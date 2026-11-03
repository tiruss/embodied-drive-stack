"""
Demo 1: Interactive 3D MuJoCo Viewer (WSLg GUI)
================================================
양팔 로봇(ALOHA ViperX 2대)이 있는 MuJoCo 3D 물리 시뮬레이션 창을 윈도우 화면에 직접 띄웁니다.
- 마우스 좌클릭 드래그 : 카메라 회전
- 마우스 우클릭 드래그 : 카메라 이동 (패닝)
- 마우스 휠 스크롤     : 줌 인/아웃
- 더블 클릭 + Ctrl+우클릭 드래그 : 로봇 팔이나 빨간 큐브에 직접 외력(힘) 가하기!
"""
import math
import time
import mujoco
import mujoco.viewer
import gymnasium as gym
import gym_aloha


def main():
    print("=" * 65)
    print("MuJoCo 3D Interactive Viewer 시작 중 (WSLg 창이 열립니다)...")
    print("종료하려면 뷰어 창을 닫거나 터미널에서 Ctrl+C를 누르세요.")
    print("=" * 65)

    # gym_aloha 환경에서 내부 MuJoCo physics 모델(model, data) 가져오기
    env = gym.make("gym_aloha/AlohaTransferCube-v0")
    env.reset()
    dm_env = env.unwrapped._env
    mj_model = dm_env.physics.model.ptr
    mj_data = dm_env.physics.data.ptr

    # 초기 관절 위치 저장
    init_ctrl = mj_data.ctrl.copy()

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        # 보기 좋은 카메라 각도 설정
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -30.0
        viewer.cam.distance = 1.4
        viewer.cam.lookat[:] = [0.0, 0.0, 0.25]

        t0 = time.time()
        while viewer.is_running():
            step_start = time.time()
            t = step_start - t0

            # 양팔 로봇이 부드럽게 인사하듯 움직이는 사인파(Sine wave) 데모 제어 입력
            ctrl = init_ctrl.copy()
            # 왼쪽 팔(0~6), 오른쪽 팔(7~13)
            ctrl[0] = 0.25 * math.sin(1.5 * t)          # 왼쪽 허리 회전
            ctrl[1] = -0.5 + 0.2 * math.sin(2.0 * t)    # 왼쪽 어깨
            ctrl[6] = 0.02 + 0.02 * math.sin(3.0 * t)   # 왼쪽 그리퍼 열기/닫기

            ctrl[7] = -0.25 * math.sin(1.5 * t)         # 오른쪽 허리 회전
            ctrl[8] = -0.5 + 0.2 * math.cos(2.0 * t)    # 오른쪽 어깨
            ctrl[13] = 0.02 + 0.02 * math.cos(3.0 * t)  # 오른쪽 그리퍼 열기/닫기

            mj_data.ctrl[:] = ctrl
            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()

            # 물리 시뮬레이션 타임스텝에 맞춰 대기
            elapsed = time.time() - step_start
            if elapsed < mj_model.opt.timestep:
                time.sleep(mj_model.opt.timestep - elapsed)

    env.close()


if __name__ == "__main__":
    main()
