"""xArm SDK boundary: base positions in meters, orientation in RPY radians.

The RobotWorker owns this controller. No automatic mock fallback or error clear.
"""
from pathlib import Path
from types import SimpleNamespace
from collections import deque
import json
import threading
import time
import numpy as np
import yaml
from system.shared_state import RobotState, ROBOT_CMD_STOP, ROBOT_CMD_HOLD, ROBOT_CMD_MOVE_TO_POSITION, ROBOT_CMD_SERVO_TO_POSITION, GRIPPER_OPEN, GRIPPER_CLOSE


class GripperCommunicationError(RuntimeError):
    """A failed tool transaction must not trigger another gripper command."""


class GripperActivationLostError(RuntimeError):
    """Unexpected runtime deactivation must not start an activation motion."""


def read_reported_offsets(arm, *, timeout_s=5.0):
    """Wait for fresh SDK reports before reading asynchronously cached offsets.

    A report callback runs before the rich report's world-offset fields are
    updated. A second callback guarantees that the previous full rich report
    has finished, including both TCP and world offsets.
    """
    ready = threading.Event()
    reports = 0

    def on_report(_report):
        nonlocal reports
        reports += 1
        if reports >= 2:
            ready.set()

    try:
        if not arm.register_report_callback(on_report):
            raise RuntimeError('Could not register xArm status report callback')
        if not ready.wait(timeout_s):
            raise TimeoutError(
                f'xArm status reports not received within {timeout_s:g}s; '
                'TCP/world offsets cannot be validated'
            )
        offsets = {}
        for name in ('tcp_offset', 'world_offset'):
            value = np.asarray(getattr(arm, name), dtype=float)
            if value.shape != (6,) or not np.isfinite(value).all():
                raise RuntimeError(f'Invalid reported xArm {name}: {value}')
            offsets[name] = value.tolist()
        return offsets
    finally:
        arm.release_report_callback(on_report)


def load_teach(config, config_path):
    path=Path(config['robot']['teach_file']).expanduser()
    if not path.is_absolute(): path=Path(config_path).resolve().parent.parent/path
    data=yaml.safe_load(path.read_text()) or {}
    place_key='place_object_xyz_mm' if 'place_object_xyz_mm' in data else 'place_object_xy_mm'
    for key,n in [('home_joints_rad',7),(place_key,3 if place_key=='place_object_xyz_mm' else 2),('approach_offset_mm',3)]:
        value=data.get(key)
        if not isinstance(value,list) or len(value)!=n or not np.isfinite(value).all():
            raise ValueError(f'Teach {key} requires {n} finite values: {path}')
    bounds=data.get('workspace_bounds_m')
    if not isinstance(bounds,dict): raise ValueError('Teach workspace_bounds_m is required')
    for axis in 'xyz':
        value=bounds.get(axis)
        if not isinstance(value,list) or len(value)!=2 or not np.isfinite(value).all() or value[0]>=value[1]:
            raise ValueError(f'Invalid teach workspace {axis}')
    if not isinstance(data.get('backoff_x_mm'),(int,float)) or not np.isfinite(data['backoff_x_mm']): raise ValueError('Teach backoff_x_mm is required')
    for axis,position in zip('xyz',data[place_key]):
        if not bounds[axis][0] <= position/1000 <= bounds[axis][1]: raise ValueError(f'Place {axis.upper()} outside teach workspace')
    if place_key=='place_object_xyz_mm':
        data['place_object_xy_mm']=list(data[place_key][:2])
        descend=data.get('place_descend_mm',0.0)
        if not isinstance(descend,(int,float)) or not np.isfinite(descend) or descend<0:
            raise ValueError('Teach place_descend_mm must be finite and nonnegative')
        release_z=(data[place_key][2]-descend)/1000.0
        if not bounds['z'][0]<=release_z<=bounds['z'][1]:
            raise ValueError('Place descent target Z outside teach workspace')
    return data


class XArmController:
    def __init__(self,config,*,sdk=None):
        self.config=config
        self.arm=sdk
        cfg=config['robot']['xarm']
        self.robot_ip=cfg['robot_ip']
        self.fixed_orientation_format='rpy'
        self._mode=None
        self._connected=False
        self._previous_pose=None
        self._previous_time=None
        self._last_gripper_state=None
        self._servo_deadline=None
        self.watchdog_timeout=float(cfg.get('watchdog_timeout_sec',0.5))
        self.gripper_position_complete_threshold=int(config['robot']['gripper'].get('position_complete_threshold',200))
        self.last_debug=SimpleNamespace(last_command_type=None,watchdog_triggered=False)
        self.using_mock=False
        comm=config['robot']['gripper'].get('communication',{})
        self.gripper_baudrate=int(comm.get('baudrate',115200))
        self.gripper_modbus_timeout_ms=int(comm.get('modbus_timeout_ms',500))
        self._gripper_min_interval=float(comm.get('min_interval_sec',0.1))
        self._gripper_command_settle=float(comm.get('command_settle_sec',0.2))
        self._gripper_reset_settle=float(comm.get('reset_settle_sec',0.5))
        self._gripper_activate_settle=float(comm.get('activate_settle_sec',2.0))
        self._gripper_baud_settle=float(comm.get('baud_settle_sec',1.0))
        delays=[self._gripper_min_interval,self._gripper_command_settle,self._gripper_reset_settle,self._gripper_activate_settle,self._gripper_baud_settle]
        if self.gripper_baudrate<=0 or self.gripper_modbus_timeout_ms<=0 or not np.isfinite(delays).all() or min(delays)<0:
            raise ValueError('Invalid robot.gripper.communication settings')
        self._gripper_io_lock=threading.RLock()
        self._next_gripper_io_t=0.0
        self._gripper_link_error=None
        self._gripper_runtime_started=False
        self._gripper_activation_error=None
        self._gripper_phase='idle'
        self._gripper_last_io_end=None
        self._gripper_io_history=deque(maxlen=20)

    @property
    def gripper_communication_error(self):
        return self._gripper_link_error

    def _configure_gripper_communication(self):
        """Configure the tool port once, before enabling robot motion."""
        baud=int(self._check(self.arm.get_rs485_baudrate(target='robot'),'get_rs485_baudrate'))
        self._check(self.arm.set_checkset_default_baud(3,self.gripper_baudrate),'set_checkset_default_baud')
        if baud!=self.gripper_baudrate:
            print(f'[INFO] Setting tool RS485 baudrate {baud} -> {self.gripper_baudrate} before robot enable.')
            self._check(self.arm.set_rs485_baudrate(self.gripper_baudrate,target='robot'),'set_rs485_baudrate')
            time.sleep(self._gripper_baud_settle)
        actual=int(self._check(self.arm.get_rs485_baudrate(target='robot'),'verify RS485 baudrate'))
        if actual!=self.gripper_baudrate:
            raise RuntimeError(f'Robotiq baudrate mismatch: actual={actual}, expected={self.gripper_baudrate}')
        self._check(self.arm.set_rs485_timeout(self.gripper_modbus_timeout_ms,target='robot',protocol='modbus_rtu'),'set_rs485_timeout')
        timeout=int(self._check(self.arm.get_rs485_timeout(target='robot',protocol='modbus_rtu'),'get_rs485_timeout'))
        if timeout!=self.gripper_modbus_timeout_ms:
            raise RuntimeError(f'Robotiq Modbus timeout mismatch: actual={timeout}ms, expected={self.gripper_modbus_timeout_ms}ms')
        errors=self._check(self.arm.get_err_warn_code(),'verify errors after tool communication setup')
        if any(errors):
            raise RuntimeError(f'xArm error/warning after Robotiq link setup: {errors}')
        print(f'[INFO] Robotiq tool-port settings verified: baud={actual}, Modbus timeout={timeout}ms, minimum interval={self._gripper_min_interval:.3f}s, command settle={self._gripper_command_settle:.3f}s')

    def _gripper_io(self,operation,function,*,settle_s=0.0,request=None):
        with self._gripper_io_lock:
            if self._gripper_activation_error is not None:
                raise self._gripper_activation_error
            if self._gripper_link_error is not None:
                raise self._gripper_link_error
            delay=self._next_gripper_io_t-time.monotonic()
            if delay>0:time.sleep(delay)
            self._healthy()
            started=time.monotonic()
            gap=None if self._gripper_last_io_end is None else started-self._gripper_last_io_end
            entry={'phase':self._gripper_phase,'operation':operation,'timestamp':time.time(),
                   'gap_sec':gap,'previous_status':dict(getattr(self.arm,'robotiq_status',{}))}
            if request is not None:
                entry['request']=dict(request)
            result=None
            try:
                result=function()
                if isinstance(result,(tuple,list)):
                    entry['code']=result[0]
                    if len(result)>1 and isinstance(result[1],(list,tuple,bytes,bytearray)):
                        entry['response']=list(result[1])
                        entry['response_length']=len(result[1])
                else:
                    entry['code']=result
                self._check(result,operation)
                entry['elapsed_sec']=time.monotonic()-started
                self._gripper_io_history.append(entry)
            except Exception as exc:
                entry['elapsed_sec']=time.monotonic()-started
                entry['error']=str(exc)
                self._gripper_io_history.append(entry)
                try:
                    errors=self._check(self.arm.get_err_warn_code(),'gripper error diagnostics')
                    arm_error=int(errors[0])
                except Exception:
                    arm_error=None
                port_settings={}
                # Controller settings only: do not send another transaction
                # to the failed gripper or clear the robot's error.
                for key,read in (
                    ('baudrate',lambda:self.arm.get_rs485_baudrate(target='robot')),
                    ('timeout_ms',lambda:self.arm.get_rs485_timeout(target='robot',protocol='modbus_rtu')),
                ):
                    try:
                        port_settings[key]=self._check(read(),f'diagnostic {key}')
                    except Exception as read_exc:
                        port_settings[key]=str(read_exc)
                message=(f'Robotiq communication failed during {operation}: {exc}; '
                         f'phase={self._gripper_phase}, response_length={entry.get("response_length")}, '
                         f'previous_io_gap_s={gap}, '
                         f'xArm error={"C19" if arm_error==19 else arm_error}, '
                         f'baud={self.gripper_baudrate}, Modbus timeout={self.gripper_modbus_timeout_ms}ms, '
                         f'elapsed={entry["elapsed_sec"]:.3f}s, port_readback={port_settings}. '
                         'Further gripper transactions are disabled until reconnect.')
                self._gripper_link_error=GripperCommunicationError(message)
                print(f'[ERROR] {message}',flush=True)
                print('[ERROR] Robotiq recent transactions: '+json.dumps(list(self._gripper_io_history),default=str),flush=True)
                raise self._gripper_link_error from exc
            finally:
                self._gripper_last_io_end=time.monotonic()
                self._next_gripper_io_t=self._gripper_last_io_end+max(self._gripper_min_interval,settle_s)
            return result

    @classmethod
    def from_config(cls,path):
        config=yaml.safe_load(Path(path).read_text())
        teach=load_teach(config,path)
        config['safety']['workspace_bounds_m']=teach['workspace_bounds_m']
        return cls(config)

    @staticmethod
    def _check(result,operation):
        code=result[0] if isinstance(result,(list,tuple)) else result
        if code != 0: raise RuntimeError(f'xArm {operation} failed: code={code}')
        return result[1] if isinstance(result,(list,tuple)) and len(result)>1 else None

    def connect(self):
        if self.arm is None:
            from xarm.wrapper import XArmAPI
            self.arm=XArmAPI(self.robot_ip,is_radian=True,default_robotiq_baud=self.gripper_baudrate,report_type='rich')
        try:
            cfg=self.config['robot']['xarm']
            offsets=read_reported_offsets(self.arm,timeout_s=float(cfg.get('report_timeout_sec',5.0)))
            errors=self._check(self.arm.get_err_warn_code(),'get_err_warn_code')
            if errors[0]==19:
                raise GripperCommunicationError('xArm C19 is already active. Clear the tool communication fault in Studio after checking the link, then reconnect; no motion or gripper commands were sent.')
            if any(errors): raise RuntimeError(f'Robot has error/warning: {errors}')
            for name,key in [('tcp_offset','expected_tcp_offset_mm_rad'),('world_offset','expected_world_offset_mm_rad')]:
                expected=cfg.get(key)
                actual=offsets[name]
                if expected is not None and not np.allclose(actual,expected,atol=1e-3,rtol=0):
                    raise RuntimeError(f'{name} differs from configured handover setup: actual={actual}, expected={expected} (XYZ mm, RPY rad)')
            print(f'[INFO] xArm reported offsets verified: {offsets} (XYZ mm, RPY rad)')
            self._configure_gripper_communication()
            self._check(self.arm.motion_enable(enable=True),'motion_enable')
            self._set_mode(0)
            self._connected=True
        except BaseException:
            self.arm.disconnect()
            raise

    def _set_mode(self,mode):
        if self._mode != mode:
            if self._mode==1: self._check(self.arm.set_state(4),'stop before position mode')
            self._check(self.arm.set_mode(mode),'set_mode')
            self._check(self.arm.set_state(0),'set_state')
            self._mode=mode

    def _healthy(self):
        if not self._connected or not self.arm.connected: raise RuntimeError('xArm disconnected')
        if self._gripper_activation_error is not None: raise self._gripper_activation_error
        errors=self._check(self.arm.get_err_warn_code(),'get_err_warn_code')
        if errors[0]==19:
            if self._gripper_link_error is None:
                self._gripper_link_error=GripperCommunicationError(
                    f'xArm C19: end-effector communication error; baud={self.gripper_baudrate}, '
                    f'Modbus timeout={self.gripper_modbus_timeout_ms}ms. '
                    'Further gripper transactions are disabled until reconnect.'
                )
            raise self._gripper_link_error
        if errors[0]: raise RuntimeError(f'xArm error: {errors[0]}')
        state=self._check(self.arm.get_state(),'get_state')
        if state==5 or (state==4 and self._mode is not None): raise RuntimeError(f'xArm stopped: state={state}')

    def _pose(self,position,orientation):
        xyz=np.asarray(position,dtype=float)
        rpy=np.asarray(orientation,dtype=float)
        if xyz.shape!=(3,) or rpy.shape!=(3,) or not np.isfinite(np.r_[xyz,rpy]).all(): raise ValueError('Finite XYZ meters and RPY radians required')
        for i,axis in enumerate('xyz'):
            low,high=self.config['safety']['workspace_bounds_m'][axis]
            if not low <= xyz[i] <= high: raise ValueError(f'Target outside workspace {axis}')
        return np.r_[xyz*1000,rpy].tolist()

    def step(self,command,now_timestamp=None):
        now=time.time() if now_timestamp is None else float(now_timestamp)
        if command.stop_requested or command.command_type==ROBOT_CMD_STOP:
            self.stop_joint_motion()
        else:
            self._healthy()
            if command.command_type in (ROBOT_CMD_MOVE_TO_POSITION,ROBOT_CMD_SERVO_TO_POSITION):
                if not command.valid or now-command.timestamp>self.watchdog_timeout or command.timestamp-now>self.watchdog_timeout:
                    self.stop_joint_motion()
                    raise RuntimeError('Stale or invalid robot command')
                pose=self._pose(command.target_position_base,command.fixed_orientation_base)
                if command.command_type==ROBOT_CMD_SERVO_TO_POSITION:
                    self._set_mode(1)
                    self._check(self.arm.set_servo_cartesian(pose,is_radian=True),'set_servo_cartesian')
                    self._servo_deadline=time.monotonic()+self.watchdog_timeout
                else:
                    self._servo_deadline=None
                    self._set_mode(0)
                    speed=self.config['robot']['motion'].get('approach_speed_mps',0.1)*1000
                    accel=self.config['robot']['xarm'].get('move_acceleration',0.75)*1000
                    self._check(self.arm.set_position(*pose,speed=speed,mvacc=accel,is_radian=True,wait=False),'set_position')
            elif command.command_type != ROBOT_CMD_HOLD: raise ValueError(f'Unknown robot command {command.command_type}')
            if command.gripper_action==GRIPPER_OPEN:
                if not self.open_gripper_blocking():
                    raise RuntimeError('Robotiq open was not confirmed')
            elif command.gripper_action==GRIPPER_CLOSE: self.start_gripper_close()
        self.last_debug.last_command_type=command.command_type
        return self.read_robot_state(now_timestamp=now)

    def move_to_joint_positions(self,joints,*,speed_rad_s,acceleration_rad_s2,async_move=True):
        joints=np.asarray(joints,dtype=float)
        if joints.shape!=(7,) or not np.isfinite(joints).all(): raise ValueError('xArm7 requires seven finite joints')
        self._healthy()
        self._servo_deadline=None
        self._set_mode(0)
        self._check(self.arm.set_servo_angle(angle=joints.tolist(),speed=speed_rad_s,mvacc=acceleration_rad_s2,is_radian=True,wait=not async_move),'set_servo_angle')
        return True

    def stop_joint_motion(self):
        self._servo_deadline=None
        if self.arm is not None and self.arm.connected:
            self._check(self.arm.set_state(4),'stop')
        self._mode=None
        return True

    def isSteady(self):
        return not bool(self.arm.get_is_moving())

    def heartbeat(self):
        self._healthy()
        if self._servo_deadline is not None: self._servo_deadline=time.monotonic()+self.watchdog_timeout

    def read_robot_state(self,now_timestamp=None):
        now=time.time() if now_timestamp is None else float(now_timestamp)
        if self._servo_deadline is not None and time.monotonic()>self._servo_deadline:
            self.stop_joint_motion()
            self.last_debug.watchdog_triggered=True
        try:
            self._healthy()
            pose=np.asarray(self._check(self.arm.get_position(is_radian=True),'get_position'),float)
            pose[:3]/=1000
            joints=self._check(self.arm.get_servo_angle(is_radian=True),'get_servo_angle')
            speed=None
            if self._previous_pose is not None and now>self._previous_time:
                speed=(pose-self._previous_pose)/(now-self._previous_time)
                speed[3:]=(np.mod(pose[3:]-self._previous_pose[3:]+np.pi,2*np.pi)-np.pi)/(now-self._previous_time)
            self._previous_pose=pose.copy();self._previous_time=now
            return RobotState(actual_tcp_pose_base=tuple(pose),actual_tcp_speed=None if speed is None else tuple(speed),joint_positions=tuple(joints),is_connected=True,valid=True,robot_mode=str(self._mode),gripper_state=self._last_gripper_state,timestamp=now)
        except Exception as exc:
            return RobotState(is_connected=bool(self._connected and self.arm.connected),valid=False,last_error=str(exc),timestamp=now)

    def _gripper(self,position,*,opening=False):
        cfg=self.config['robot']['gripper'];prefix='open' if opening else 'close'
        position=int(position);speed=int(cfg.get(prefix+'_speed',255));force=int(cfg.get(prefix+'_force',50))
        self._gripper_io('robotiq_set_position',lambda:self.arm.robotiq_set_position(position,speed=speed,force=force,wait=False,wait_motion=False),settle_s=self._gripper_command_settle,request={'position':position,'speed':speed,'force':force})
        self._last_gripper_state=GRIPPER_OPEN if opening else GRIPPER_CLOSE
        return True

    def start_gripper_close(self):
        self._gripper_phase='close'
        result=self._gripper(255)
        self._gripper_runtime_started=True
        return result

    def _read_gripper_status(self):
        with self._gripper_io_lock:
            self._gripper_io('robotiq_get_status',self.arm.robotiq_get_status)
            status=dict(self.arm.robotiq_status)
        required={'gPO','gOBJ','gFLT','gACT','gSTA'}
        if not required.issubset(status):
            raise RuntimeError(f'Robotiq parsed status missing fields: {sorted(required-set(status))}')
        if self._gripper_runtime_started and (status['gACT']!=1 or status['gSTA']!=3 or status['gFLT']==5):
            message=(f'Robotiq activation/readiness lost during runtime: phase={self._gripper_phase}, status={status}. '
                     'Automatic reset/reactivation is disabled after startup; stopping the task. '
                     'Check power/connection and other gripper controllers before reconnecting. '
                     'This status alone does not prove a power loss.')
            self._gripper_activation_error=GripperActivationLostError(message)
            print(f'[ERROR] {message}',flush=True)
            print('[ERROR] Robotiq recent transactions: '+json.dumps(list(self._gripper_io_history),default=str),flush=True)
            raise self._gripper_activation_error
        return status

    def ensure_gripper_ready(self):
        """Activate on startup only; runtime deactivation is latched by the reader."""
        cfg=self.config['robot']['gripper']
        timeout=float(cfg.get('activation_timeout_sec',10.0))
        deadline=time.monotonic()+timeout
        status=self._read_gripper_status()
        if status['gFLT'] not in (0,5,9):
            raise RuntimeError(f'Robotiq fault {status["gFLT"]}: {status}')
        # The SDK also regards gSTA=3 with fault 9 as activated: idle
        # communication timeout is cleared by resuming gripper commands.
        if status['gACT']==1 and status['gSTA']==3 and status['gFLT'] in (0,9):
            return status
        if not (status['gACT']==1 and status['gSTA']==1):
            if not cfg.get('activate_on_connect',False):
                raise RuntimeError(
                    f'Robotiq is not activated: {status}; '
                    'enable robot.gripper.activate_on_connect to initialize it'
                )
            print(f'[INFO] Activating Robotiq before startup open: {status}')
            self._gripper_io('robotiq_reset',self.arm.robotiq_reset,settle_s=self._gripper_reset_settle)
            while time.monotonic()<deadline:
                status=self._read_gripper_status()
                if status['gFLT'] not in (0,5,9):
                    raise RuntimeError(f'Robotiq reset fault {status["gFLT"]}: {status}')
                if status['gACT']==0 and status['gSTA']==0:
                    break
                time.sleep(0.05)
            else:
                raise TimeoutError(f'Robotiq reset timeout after {timeout:g}s: {status}')
            self._gripper_io('robotiq_set_activate',lambda:self.arm.robotiq_set_activate(wait=False),settle_s=self._gripper_activate_settle)
        else:
            print('[INFO] Waiting for Robotiq activation already in progress...')
        while time.monotonic()<deadline:
            status=self._read_gripper_status()
            if status['gFLT'] not in (0,5,9):
                raise RuntimeError(f'Robotiq activation fault {status["gFLT"]}: {status}')
            if status['gACT']==1 and status['gSTA']==3 and status['gFLT'] in (0,9):
                print('[INFO] Robotiq activation complete')
                return status
            time.sleep(0.05)
        raise TimeoutError(f'Robotiq activation timeout after {timeout:g}s: {status}')

    def open_gripper_blocking(self):
        cfg=self.config['robot']['gripper']
        prepare_delay=float(cfg.get('open_prepare_delay_sec',0.0))
        if not np.isfinite(prepare_delay) or prepare_delay<0:
            raise ValueError('open_prepare_delay_sec must be finite and nonnegative')
        self._gripper_phase='open_prepare'
        # Existing communication errors must fail immediately without waiting.
        if self._gripper_link_error is not None:
            raise self._gripper_link_error
        if self._gripper_activation_error is not None:
            raise self._gripper_activation_error
        print(f'[INFO] Robotiq open prepare: settling {prepare_delay:g}s before status read.',flush=True)
        time.sleep(prepare_delay)
        status=self.ensure_gripper_ready()
        print(f'[INFO] Robotiq open requested: target=0, before={status}',flush=True)
        self._gripper_phase='open_command'
        self._gripper(0,opening=True)
        timeout=float(self.config['robot']['gripper'].get('open_timeout_sec',5.0))
        deadline=time.monotonic()+timeout
        activation_retried=False
        self._gripper_phase='open_wait'
        while time.monotonic()<deadline:
            status=self._read_gripper_status()
            if not {'gPR','gGTO'}.issubset(status):
                raise RuntimeError(f'Robotiq open status missing command echo gPR/gGTO: {status}')
            fault=status['gFLT']
            if fault not in (0,5,9):
                raise RuntimeError(f'Robotiq open fault {fault}: {status}')
            if fault==5 and not activation_retried:
                # Activation may consume the first go-to request after idle.
                # Confirm activation before retrying that open request once.
                print(f'[INFO] Robotiq open delayed by activation; checking readiness: {status}')
                self.ensure_gripper_ready()
                self._gripper_phase='open_command_after_activation'
                self._gripper(0,opening=True)
                self._gripper_phase='open_wait'
                activation_retried=True
                deadline=time.monotonic()+timeout
                continue
            accepted=fault==0 and status['gACT']==1 and status['gSTA']==3 and status['gPR']==0 and status['gGTO']==1
            if accepted and status['gOBJ']==1:
                raise RuntimeError(f'Robotiq open blocked by contact (gOBJ=1); release unconfirmed: {status}')
            if accepted and status['gPO']<=5 and status['gOBJ']==3:
                print(f'[INFO] Robotiq open confirmed: {status}',flush=True)
                self._gripper_phase='open_complete'
                self._gripper_runtime_started=True
                return True
            time.sleep(0.02)
        # Leave the open target intact. A hold/go-to here could replace slow
        # opening with a partly closed target. The caller aborts arm retreat.
        raise TimeoutError(f'Robotiq open timeout: {status}')

    def get_gripper_close_state(self):
        status=self._read_gripper_status()
        if status.get('gFLT',0): raise RuntimeError(f'Robotiq fault {status["gFLT"]}')
        if 'gPO' not in status or 'gOBJ' not in status: raise RuntimeError('Robotiq parsed status missing gPO/gOBJ')
        names={0:'MOVING',1:'STOPPED_OUTER_OBJECT',2:'STOPPED_INNER_OBJECT',3:'AT_DEST'}
        return {'position':int(status['gPO']),'closed_position':255,'object_status':names[int(status['gOBJ'])]}

    def stop_gripper_motion(self):
        state=self.get_gripper_close_state()
        if state['object_status'] in {'AT_DEST','STOPPED_INNER_OBJECT','STOPPED_OUTER_OBJECT'}:
            print(f'[INFO] Robotiq already stopped: {state}; preserving grasp without another position command.',flush=True)
            return True  # Already stopped; do not issue a redundant go-to.
        return self._gripper(state['position'])

    def close(self):
        try:
            self.stop_joint_motion()
        finally:
            if self.arm is not None:self.arm.disconnect()
            self._connected=False
