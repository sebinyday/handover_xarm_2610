# xArm7 SAM2 handover

듀얼 RealSense의 **SAM2 point-prompt 추적 + MediaPipe 손 추적**을 xArm7/Robotiq handover에 연결한 버전입니다. SAM3D가 생성한 canonical template을 관측 depth로 scale 초기화하고 silhouette 보정·ICP fitting 후 grasp target을 계산합니다. ROS2 없이 Python SDK를 사용합니다.

기존 ICRA의 follow→grasp→return/place, AnySkin grasp/release, hand-relative fallback, prediction, 영상·촉각·메타데이터 기록을 이식했습니다. YOLO, FastSAM, HOI-DETR, UR 통신, fill-level 추정 및 이전 보정 파일은 포함하지 않습니다.

## 준비

**SAM2·SAM3D·MediaPipe·로봇 제어 모두 `conda activate handover` 환경 하나를 사용합니다.** Python 3.10, PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121로 맞췄습니다. SAM3D 서버는 모델을 메모리에 유지하기 위한 별도 프로세스이며, 같은 환경의 Python으로 실행됩니다.

현재 환경 세팅은 완료했습니다. 일반 터미널에서 다음과 같이 확인합니다.

```bash
conda activate handover
python tools/smoke_sam2.py --frames 3
python tools/smoke_sam3d.py
python tools/smoke_sam3d.py --with-weights
python -m unittest discover -s tests -v
```

`smoke_sam3d.py`의 기본 실행은 pipeline import 및 PyTorch3D·Kaolin·spconv CUDA 연산만 검사합니다. 이때 `check_scope: dependencies_only`, `weights_loaded: false`는 이번 검사에서 가중치를 로드하지 않았다는 뜻이며, 가중치가 없거나 로딩에 실패했다는 뜻이 아닙니다. `--with-weights`를 추가하면 실제 stage1 가중치를 GPU에 로드하고 `ss_generator`·`ss_decoder`만 초기화되었는지 확인합니다. 성공하면 `check_scope: dependencies_and_weights`, `weights_loaded: true`를 출력합니다. 두 모드 모두 카메라·로봇 연결이나 template 추론은 수행하지 않습니다.

SAM2의 선택적 CUDA 후처리 확장과 stage1에서 사용하지 않는 gsplat 렌더러는 빌드하지 않았습니다. SAM2 GPU 추적과 SAM3D stage1 의존성 GPU 연산은 RTX A6000에서 검증했습니다. 실행 격리 안에서는 CUDA 접근이 제한될 수 있습니다.

추가 설치 목록은 [requirements-handover.txt](requirements-handover.txt), SAM3D 추론 의존성과 고정 소스 커밋은 [environment/sam3d-runtime-requirements.txt](environment/sam3d-runtime-requirements.txt)에 기록했습니다. 재설치 및 가중치 준비는 [external/README.md](external/README.md)를 참고하세요. 모델·카메라 경로는 [configs/handover.yaml](configs/handover.yaml)에 있습니다.

**사용자가 공유한 Google Drive 체크포인트를 다운로드해 stage1 가중치 배치를 완료했습니다.** `external/sam-3d-objects/checkpoints/hf/`에 `ss_generator`, `ss_decoder`와 관련 설정만 배치했습니다. 공식 예제 RGB·mask로 실제 모델 서버 로딩 및 stage1 template 생성을 검증했으며, 결과는 [template.npy](output/sam3d_checkpoint_validation/template.npy)와 [validation.json](output/sam3d_checkpoint_validation/validation.json)에 저장했습니다. Hugging Face 접근 승인을 기다리지 않고 현재 파일로 실행할 수 있습니다.

보정 JSON은 원본 위치에서 읽으며 복사하지 않습니다.

| 파이프라인 | 보정 결과 | Serial |
|---|---|---|
| cam0 | `xarm7_charuco_calibration/results/cam1.json` | `238222073927` |
| cam1 | `xarm7_charuco_calibration/results/cam2.json` | `335522072904` |

`T_base_color`와 `color_K`를 사용합니다. depth는 color에 정렬한 뒤 센서 depth scale로 m로 변환합니다. 활성 stream의 serial·내부 파라미터가 보정 결과와 다르면 중단합니다. 기본 해상도는 640×480이며 다른 profile을 쓰려면 일치하는 새 보정값이 필요합니다. 현재 pinhole lifting은 저장된 distortion이 0인 경우를 지원합니다.

다음 점검은 카메라·로봇·시리얼 장치를 열지 않습니다.

```bash
python tools/check_runtime.py
python tools/check_runtime.py --robot
python tools/smoke_sam2.py --cpu
python -m unittest discover -s tests -v
```

## 실행

`handover` 환경이 활성화된 터미널에서 서버를 시작합니다. 스크립트는 설정에 지정된 동일 `handover` 환경의 Python을 실행합니다.

```bash
bash run_sam3d_model_server.sh
```

`model ready` 이후, 다른 터미널에서도 `conda activate handover`를 실행하고 perception-only를 시작합니다.

```bash
bash run_handover.sh
```

각 카메라에서 **같은 물체**를 선택합니다. 좌클릭=물체, 우클릭=배경 제외, `C`=포인트 지우기, `Enter`=mask 확정입니다. 클릭 중에는 해당 RGB·depth 프레임을 고정합니다. 두 시점의 포인트를 독립적으로 선택하며 SAM2 weights는 공유하고 시간 메모리는 분리합니다.

cam0의 안정된 RGB·mask로 SAM3D template을 비동기 생성합니다. 생성·scale 초기화·연속 silhouette fitting이 끝날 때까지 로봇 target을 gate합니다. 실패 및 이전 task의 늦은 결과는 target을 활성화하지 않습니다. 한 카메라 추적만 유효해도 처리를 계속하고, 두 카메라 모두 잃으면 설정된 fallback/prediction 한도 이후 정지하고 재선택합니다.

로봇을 사용하려면 [configs/teach.yaml](configs/teach.yaml)에 **현재 설치 환경에서 확인한** 7축 HOME, place XYZ, workspace, approach vector, backoff 값을 입력해야 합니다. 비어 있으면 `--enable-follow`는 장치 연결 전에 거부됩니다. calibration 때의 로봇 자세는 HOME으로 사용하지 않습니다. `robot.xarm`의 TCP/world offset도 현재 handover 툴과 일치시켜야 하며, 연결 시 읽어서 검사합니다.

```bash
bash run_handover.sh --enable-follow --follow-z --record-video --3d-debug --profile-runtime
```

로봇 모드에서는 AnySkin이 기본 활성화됩니다. perception-only는 촉각 장치를 열지 않습니다. 모든 arm/gripper SDK 명령은 단일 RobotWorker가 소유합니다. 내부 pose는 xArm base의 m·RPY rad이고, SDK 경계에서 위치만 mm로 바꿉니다. UR의 XY 부호 반전은 적용하지 않습니다.

로봇 연결 후 SDK의 rich 상태 보고를 기다린 뒤 TCP/world offset을 검사합니다. 연결 직후의 SDK 캐시에는 아직 0이 들어 있을 수 있으므로 즉시 비교하지 않습니다. 대기 시간은 `robot.xarm.report_timeout_sec`(기본 5초)입니다. 실제 불일치가 있으면 `actual`·`expected` 값을 출력하고 초기화를 중단합니다. 초기화나 HOME 이동이 실패하면 follow도 시작하지 않습니다.

현재 박스에 놓기 설정은 `configs/teach.yaml`의 다음 값입니다.

```yaml
place_object_xyz_mm: [600.0, -300.0, 300.0]
place_descend_mm: 100.0
```

세 값 모두 xArm base 기준 **그리퍼 TCP 위치(mm)**입니다. 박스 바닥이나 물체 바닥 좌표가 아닙니다. 이 키가 있으면 먼저 지정 XYZ에 도착한 뒤 `place_descend_mm`만큼 Z로 내려가고, 하강 도착을 확인한 후 그리퍼를 엽니다. 현재 경로는 `[600,-300,300]` → `[600,-300,200]`mm → 열기입니다. `place_descend_mm`을 0으로 설정하거나 생략하면 추가 하강하지 않습니다. 물체 형상 기반 place Z 계산·샘플 대기, grasp XY offset 보정, 촉각 기반 release 하강, 고정 pre-release 하강 및 post-release Z 하강을 생략합니다. grasp 과정의 SAM3D/ICP와 AnySkin 접촉 판단은 유지합니다. 열림 확인 후 현재 `backoff_x_mm: 20`에 따라 열기 높이인 Z=200mm를 유지하면서 X=580mm로 후퇴하고 HOME으로 복귀합니다. 도착 허용 오차는 기존 `--position-tolerance-m`(기본 0.01m)을 사용합니다. 목표 도달·열림 확인에 실패하면 다음 단계를 진행하지 않습니다.

XYZ 및 하강 후 Z가 workspace를 벗어나면 시작 시 거부합니다. XYZ는 유한한 세 숫자, 하강 거리는 유한한 0 이상의 mm 값이어야 합니다. 이후 박스 위치를 바꿀 때는 이 좌표와 하강 거리를 수정하고 프로그램을 재시작합니다. 이전 `place_object_xy_mm`만 쓰는 teach 파일은 기존 형상·촉각 기반 하강 모드로 동작합니다. 두 키를 함께 쓰면 XYZ가 우선합니다. 하강 후 실제 명령한 release TCP 목표는 metadata의 `release_tcp_target_x/y/z_mm`에 기록하고, 물체 바닥 delivery 좌표는 추정하지 않습니다. 기존 metadata CSV에는 세 열을 추가하되 이전 행의 값을 보존합니다.

HOME 이동 전에 Robotiq의 parsed `gACT`·`gSTA`·`gFLT`로 준비 상태를 확인하고 그리퍼를 엽니다. 기본 `robot.gripper.activate_on_connect: true`는 미활성 상태에서만 reset 완료를 확인한 뒤 활성화합니다. 이미 활성화되었으면 유지하고, 활성화 중이면 최대 `activation_timeout_sec`(10초)까지 기다립니다. 최초 startup 열기 직후 일시적인 fault 5가 나오면 활성화 완료를 확인한 뒤 열기를 한 번만 재시도합니다. startup 열기 완료 또는 close 명령 이후에는 자동 reset·재활성화를 허용하지 않습니다. 열기 완료는 오류가 없고 활성화된 상태에서 실제 위치와 동작 완료를 확인해야 인정합니다. 영구 오류나 시간 초과는 초기화를 중단하며, 같은 명령을 무조건 반복하지 않습니다.

C19는 xArm의 말단 장치 통신 오류이며, Robotiq의 `gFLT`나 빈 닫힘의 `gOBJ=3`과 구분합니다. [UFACTORY 오류 설명](https://docs.usermanual.ufactory.cc/user_manual/ufactoryStudio/12.error_handling.html)은 연결과 baudrate 확인을 안내합니다. 시작 시 로봇 motion enable 전에 tool RS485 포트를 `115200 baud`, Modbus 응답 timeout을 `500ms`로 설정하고 읽어서 검증합니다. 이미 baudrate가 맞으면 다시 설정하지 않습니다. 이 값은 xArm 포트 설정이며, 연결된 그리퍼의 baudrate도 일치해야 합니다. 실제 그리퍼 응답은 이어지는 status 조회로 확인합니다. 로봇이 이미 C19 상태이면 명령이나 자동 오류 해제 없이 초기화를 중단합니다.

`robot.gripper.communication`에서 `baudrate`, `modbus_timeout_ms`, `min_interval_sec`(0.1초), `command_settle_sec`(0.2초), `reset_settle_sec`(0.5초), `activate_settle_sec`(2초), `baud_settle_sec`(1초)를 조정합니다. 모든 그리퍼 read/write는 응답 완료 후 최소 간격을 두고 실행하고, 위치 명령 후에는 settle 시간까지 상태 조회를 기다립니다. `500ms`는 응답을 기다릴 수 있는 최대 시간이며 성공한 모든 요청을 500ms 지연시키는 값이 아닙니다. `robot.gripper.close_timeout_sec` 기본 5초와 `--gripper-close-timeout-s`는 전체 닫기·검증 시간으로, Modbus 패킷 timeout과 별개입니다. API 명칭과 시간 단위는 [공식 SDK](https://github.com/xArm-Developer/xArm-Python-SDK/blob/master/xarm/wrapper/xarm_api.py)의 RS485 인터페이스를 사용합니다.

닫기 명령이나 status 통신에 실패하면 같은 닫기 명령을 다른 경로로 재전송하지 않고 grasp를 중단합니다. 실패한 tool transaction 이후에는 연결을 다시 만들 때까지 추가 그리퍼 read/write를 막으며, 로그에는 operation·SDK 반환 코드·xArm 오류·baudrate·timeout·소요 시간을 남깁니다. 정상적으로 닫힌 뒤 내부 물체 접촉 증거가 없으면 grasp 실패로 처리하고 불필요한 hold 위치 명령을 보내지 않습니다. `MOVING` 상태의 위치 threshold·일시적인 encoder 정지만으로 성공 처리하지 않으며, AnySkin/force의 실제 접촉 판단은 유지합니다. 이미 발생한 C19는 연결·전원·baudrate 상태를 확인하고 Studio에서 오류를 해제한 뒤 프로그램을 재시작합니다. 이후 실기 로그에서 내려놓기 중 C19 재발이 확인되었습니다. 아래 release 진단 보완도 mock 검증이며 실물 통신 문제의 해결 여부는 별도 확인이 필요합니다.

추가 실기 로그에서는 닫기 때 `gACT=1/gSTA=3`였지만, release 준비 조회가 `[9,3,6,0,0,0,0,0,0]`을 반환한 뒤 기존 코드가 `robotiq_reset`을 보내고 C19가 발생했습니다. 이 응답이 실제 장치 리셋인지 통신 경로의 이상 응답인지는 미확정입니다. 이제 작업 중 `gACT!=1`, `gSTA!=3` 또는 fault 5이면 `GripperActivationLostError`를 유지하여 추가 그리퍼 명령과 로봇 이동을 막습니다. 최초 연결의 비활성 그리퍼 초기화는 유지하며, 활성 상태가 유지된 idle fault 9도 기존대로 처리합니다. 이 변경은 자동 reset 경로를 수정한 것이며 전부 0 응답의 근본 원인을 해결했다고 검증한 것은 아닙니다.

정상 시작 로그에는 `xArm reported offsets verified` → `Robotiq startup open complete; proceeding to HOME` → `Moving robot to HOME joints` → `HOME joints reached`가 나타납니다. 인식 초기화 후 `Reference object position locked`가 출력되면, 기준 위치에서 물체를 XY 또는 Z 방향으로 50mm 이상 움직여야 추종이 시작됩니다. 목표는 `teach.yaml`의 xArm base workspace 범위로 제한되므로 그 범위 안에서 확인하세요.

내려놓을 때 손가락이 전혀 안 열리고 `robotiq_get_status: code=23; xArm error=C19`가 나오면 status 통신 실패로 이후 명령이 차단된 것입니다. SDK의 code 23은 Modbus 응답 길이 오류이며, 이 로그만으로 baudrate 불일치나 대기 시간 부족을 확정할 수 없습니다. `elapsed=1.501s`는 SDK 호출 전체 시간이며 `500ms` 설정의 실제 적용 여부나 재시도 횟수를 뜻하지 않습니다. `open_prepare_delay_sec: 0.5`를 추가한 뒤에도 마지막 통신으로부터 1.755초 후 `open_prepare` 첫 조회에서 빈 응답과 C19가 재발했습니다. 추가 대기는 기본값 `0.0`으로 되돌렸으며, baudrate·Modbus timeout·기존 transaction 간격은 유지합니다. 같은 실행의 앞부분에서는 34.297초 공백 후 닫기가 성공했으므로 공백 길이만으로 C19를 설명할 수는 없습니다. 오류 후 baudrate readback의 code 1도 실제 baudrate 값이나 불일치 증거가 아닙니다.

마지막 정상 응답은 `gOBJ=2`, `gPO=98`로 물체 접촉에 의해 이미 정지한 상태였습니다. 이전 `stop_gripper_motion()`은 이 경우에도 현재 위치로 새 go-to를 보냈으므로, 이제 접촉 정지(`gOBJ=1/2`)와 목표 도달(`gOBJ=3`)에서는 추가 위치 명령을 생략합니다. 움직이는 중(`gOBJ=0`)에는 현재 위치로 정지시키는 기존 동작을 유지합니다. 이는 불필요한 동작 명령 제거이며 C19의 직접 원인을 입증하거나 실물에서 해결을 확인한 것은 아닙니다. 최근 transaction 기록에는 새 위치 명령의 position·speed·force도 포함합니다.

열기 로그는 `RELEASE opening` → `Robotiq open prepare` → `open requested` → `open confirmed` 순서입니다. 통신 실패 로그의 `phase=open_prepare/open_command/open_wait`로 첫 상태 조회·명령 전송·완료 대기를 구분합니다. 실패 시 실제 포트 설정 readback, SDK 응답 길이, 마지막 통신 간격 및 최근 20개 transaction도 출력합니다. `previous_status`는 마지막 수신 상태로서 실패 시점의 최신 상태가 아닙니다. 열림 완료는 [Robotiq 상태 정의](https://assets.robotiq.com/website-assets/support_documents/document/online/2F-85_2F-140_TM_InstructionManual_HTML5_20190503.zip/2F-85_2F-140_TM_InstructionManual_HTML5/Content/4.%20Control.htm)에 따라 `gPR=0`, `gGTO=1`, `gOBJ=3`, 실제 위치 `gPO<=5`를 함께 확인합니다. 열리는 중 접촉으로 멈추거나 timeout이면 추가 hold 위치 명령 없이 실패 처리하며, 열림 확인 전에는 release 시각 기록·후퇴·HOME을 진행하지 않습니다. 기록의 last-contact 시각은 물리적 접촉 해제를 직접 측정한 값이 아니라 열림 확인 시각입니다. C19 이후 자동 오류 해제나 열기 재전송은 하지 않습니다.

로봇 초기화·grasp/place·HOME 복귀 중에는 point 선택과 SAM3D 이미지 캡처를 막습니다. HOME 관절 목표와 정지 상태를 확인한 뒤 두 카메라의 대기 프레임과 follower 캐시를 비우고, `runtime.prompt_capture_settle_sec`(0.2초) 동안 depth를 갱신한 후 새로운 RGB·depth 쌍으로 선택창을 엽니다. 다음 물체는 `g`로 HOME 복귀 후 template을 재생성하거나 `r`로 새 task를 시작합니다. 작업 완료 상태에서는 자동으로 선택창을 열지 않습니다.

같은 물체로 반복할 때는 `h`를 사용합니다. HOME 복귀 후 현재 실행에서 생성한 SAM3D template과 SAM2 temporal memory를 유지하고, 검증이 끝난 scale·rotation으로 ICP 위치 추종을 다시 시작합니다. 이전 place 위치와 멀어져도 첫 유효 점군의 중심에서 ICP를 시작합니다. HOME 완료를 기다린 뒤 새 RGB·depth 쌍을 받고, 연속으로 유효한 ICP가 확인되어야 로봇 목표를 허용합니다. SAM3D 생성 요청을 새로 보내지 않습니다. 최초 scale 검증이 아직 끝나지 않았다면 template만 재사용하고 기존 scale/silhouette 초기화를 완료합니다.

SAM2 마스크는 물체 점군을 만드는 데 계속 필요하지만, 추적이 유지되면 point를 다시 클릭할 필요는 없습니다. 한 시점이 유실되면 다른 시점으로 계속 처리하고, 두 시점 모두 유효하지 않거나 추적 상태가 사라지면 재선택을 받습니다. 이때도 `h`의 template은 재생성하지 않습니다. 현재 ICP는 위치만 보정하므로 물체 방향이 크게 바뀌었다면 `p`로 prompt와 초기 자세 fitting을 다시 잡습니다. `h`는 `r`처럼 현재 미저장 영상·profile을 초기화하므로 결과를 보관하려면 먼저 `s`로 저장합니다. 다른 물체에는 `g`를 사용하며, 프로그램을 재시작하면 먼저 template 생성이 필요합니다.

그리퍼를 닫는 위치 허용 오차는 `grasp.pregrasp_tolerance_mm: [10, 10, 10]`입니다. 기존 X 190mm 허용 조건을 제거했고, 물체 이동이 감지된 뒤 최신 TCP 목표의 각 축 10mm 안까지 접근해야 닫습니다. 카메라에서 구한 원래 grasp point가 아니라 approach offset이 반영된 TCP 목표와 비교합니다. 추가 삽입 거리 자체를 늘리는 설정과는 별개입니다.

촉각 release 하강 하한은 `max(40mm, robot.tactile.release_descent_min_z_mm, teach workspace Z 하한)`을 사용합니다. 현재 teach의 Z 하한 80mm를 유지하며, workspace 밖인 40mm 명령으로 중단되지 않도록 했습니다. AnySkin trigger 또는 유효한 하한 도달 이후 기존 release 흐름을 수행합니다. TCP 피드백이 유효하지 않으면 열지 않고 실패를 표시합니다. mm→m 변환은 float64를 사용해 80mm 경계가 반올림 오차로 workspace 밖으로 판정되는 것을 방지합니다.

| 키 | 동작 |
|---|---|
| `f` | follow 시작/정지 (로봇 모드) |
| `p` | follow를 멈추고 같은 물체의 두 시점 prompt 수정, fitting 재초기화 |
| `h` | 같은 물체 반복: HOME→기존 SAM3D·SAM2 상태 재사용→ICP 재피팅 |
| `g` | follow 정지→HOME 복귀→prompt 선택→SAM3D template 재생성 |
| `r` | 저장하지 않고 새 task로 reset |
| `d` | 3D debug 저장 후 reset |
| `s` | metadata·영상·profile 저장 및 follow 정지 |
| `q` / Esc | 종료 |

영상은 sensor hub 프레임을 전달받아 카메라를 중복으로 열지 않습니다. 처리 FPS가 낮으면 프레임을 반복 기록하여 동영상 길이를 wall time에 맞춥니다. 영상 web UI는 설정에서 켤 수 있습니다. 저장 결과는 `output/`, `recordings/`에 생성됩니다. 메타데이터의 수동 실험 입력은 유지하고 vision fill-level·질량 추정 필드는 제거했습니다.

## 코드 구조

- `handover.py`: 이식한 task 상태, RobotWorker, grasp/place·AnySkin, main loop.
- `pipeline.py`, `calibration_io.py`: 활성 perception 조립과 ChArUco JSON 입력.
- `perception/sam2_tracker.py`: 공식 video predictor를 live stream에 연결하는 bounded-memory adapter.
- `perception/sam3d_live_runtime.py`: 포인트 UI, stable capture, template generation epoch와 readiness.
- `robot/xarm_controller.py`: SDK mode 전환, 단위·상태·그리퍼, watchdog, 오류 처리.
- 나머지 perception/system/utils: 기존 3D 처리와 기록 모듈을 선별 이식.

SAM2 adapter는 공식 per-object inference-state API에 의존합니다. 외부 SAM2 source를 바꾸면 smoke·장시간 메모리 테스트를 다시 실행하세요. 보관 RGB는 최신 1개, conditioning output은 첫 prompt 1개, 나머지 출력은 설정된 history 범위로 제한합니다.

## 검증 범위와 출처

하드웨어를 연결하지 않은 단위·통합 테스트 **163개가 통과**했습니다. 공용 PyTorch 2.5.1+cu121 환경에서 SAM2의 두 카메라 독립 추적 상태를 각각 3프레임 GPU 검증했고, SAM3D stage1 pipeline import 및 PyTorch3D·Kaolin·spconv CUDA 연산을 검증했습니다. `smoke_sam3d.py --with-weights`로 실제 stage1 모델만 로드되는 것도 확인했습니다. 테스트에는 1,000프레임 memory 제한, frame identity, 7축 HOME, SDK 단위·mode 전환, Robotiq 활성화·열기·HOME 순서와 실패 차단, HOME 이후 fresh capture, h의 template·SAM2 재사용·ICP 재피팅·task epoch 보존, grasp 접근 허용 오차, 80mm 하한에서 release 후 HOME 복귀, timeout/cancel, template 실패·늦은 결과 차단, main loop 종료 정리와 공식 SAM3D API 호환 adapter가 포함됩니다.

공유 체크포인트로 모델 서버 초기화가 약 62.1초(최초 공개 보조 모델 다운로드 포함), stage1 추론이 약 8.15초 걸렸고 유한한 좌표의 25,819×3 template을 생성했습니다. 이는 공식 예제 이미지 1회 검증 결과입니다. 로봇에 읽기 전용으로 연결해 상태 보고 대기 후 TCP `[0, 0, 174, 0, 0, 0]`과 world offset `[0, 0, 0, 0, 0, 0]`이 설정과 일치하는 것을 확인했습니다. 실제 카메라에서의 fitting 정밀도, AnySkin 장치 연결, HOME 이동·workspace 및 로봇 handover 실기 검증은 남아 있습니다. `python tools/check_runtime.py --robot`가 설치 및 준비 상태를 표시합니다. 기존 `pyzed 5.3`의 NumPy 2 요구사항과 현재 NumPy 1.26.4의 충돌은 남아 있으며, 이 프로젝트는 ZED를 사용하지 않습니다.

ICRA 이식 기준: `723ea499cb707631dbdd3b22d88c5e72d7377b4b` (`/home/sebin/ICRA_vision_module`). 원본 MIT [LICENSE](LICENSE)를 보존했습니다. 모델 및 외부 SDK의 라이선스는 각 upstream을 따릅니다.
