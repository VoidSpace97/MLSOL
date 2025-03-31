#!/usr/bin/env python3
import subprocess
import sys
from datetime import datetime

def main():
    # Crea un run_id (se vuoi puoi metterci un timestamp più la parola "session")
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"session_{timestamp_str}"

    print(f"[LAUNCHER] Avvio inference e paper_trading con run_id={run_id}")

    # Avvia inference.py e paper_trading.py in parallelo, passandogli --run_id
    inference_proc = subprocess.Popen([sys.executable, "inference.py", "--run_id", run_id])
    paper_proc = subprocess.Popen([sys.executable, "paper_trading.py", "--run_id", run_id])

    # Attendi che i due processi terminino (o che ricevano Ctrl+C)
    inference_proc.wait()
    paper_proc.wait()

    print("[LAUNCHER] Entrambi gli script si sono interrotti.")

if __name__ == "__main__":
    main()
