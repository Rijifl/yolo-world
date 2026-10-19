r"""Pre-class check: is the laptop GPU running at full speed?

Run from the repo root:  .venv\Scripts\python scripts\gpu_check.py
Exit code 0 = GPU is healthy, 1 = use the CPU (GPU throttled, stuck or missing).
"""
import subprocess
import sys
import time

LOAD = ("import torch,time; a=torch.randn(2048,2048,device='cuda'); t=time.time()\n"
        "while time.time()-t<8: (a@a).sum().item()")
QUERY = ["nvidia-smi", "--query-gpu=name,pstate,clocks.sm,temperature.gpu,power.draw", "--format=csv,noheader,nounits"]


def main() -> int:
    try:
        import torch
        if not torch.cuda.is_available():
            print("RESULT: no CUDA GPU visible -> use the CPU")
            return 1
    except Exception as e:
        print(f"RESULT: torch failed to load ({e}) -> use the CPU")
        return 1
    load = subprocess.Popen([sys.executable, "-c", LOAD], stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(5)  # read the clock while the load is running
    try:
        out = subprocess.run(QUERY, capture_output=True, text=True, timeout=15).stdout.strip()
        name, pstate, clock, temp, power = [x.strip() for x in out.split(",")]
        clock = float(clock)
    except Exception as e:
        load.kill()
        print(f"RESULT: could not read nvidia-smi ({e}) -> use the CPU")
        return 1
    try:
        load.wait(timeout=30)
    except subprocess.TimeoutExpired:
        load.kill()
        print("RESULT: the GPU test did not finish (GPU stuck) -> use the CPU. Restart the laptop to fix it.")
        return 1
    print(f"{name}: state {pstate}, clock {clock:.0f} MHz, {temp} C, {power} W under load")
    if clock < 1000:
        print("RESULT: the GPU is THROTTLED (clock under 1000 MHz) -> use the CPU, it is faster right now.")
        print("        To fix: plug in the charger, set the power mode to best performance, or restart.")
        return 1
    print("RESULT: GPU OK -> use the GPU")
    return 0


if __name__ == "__main__":
    sys.exit(main())
