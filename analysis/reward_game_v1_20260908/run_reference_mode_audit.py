"""Diagnose reference-mode numerics after the current audio evaluation."""
import signal
import subprocess
import time
from conceptmod.textsliders.reward_game.core import DEFAULT_HOME,Store,read


def main():
    def cancel(signum,frame):raise KeyboardInterrupt(f'Cancellation signal {signum}')
    signal.signal(signal.SIGTERM,cancel)
    h=DEFAULT_HOME
    while subprocess.run(['systemctl','--user','is-active','--quiet','music-reward-game-residual-learning-v1-20260908.service']).returncode==0:
        if (h/'STOP').exists():return
        time.sleep(5)
    if (h/'STOP').exists():return
    if read(h/'status.json')['state'] not in ('method_selection_required','confirmation_required'):
        Store(h).event('reference_mode_audit_deferred',reason='Resolve the active evaluation result first')
        return
    Store(h).status('reference_mode_audit',research_complete=False)
    from conceptmod.textsliders.reward_game.audit_reference_mode import audit
    audit(h,1)
    Store(h).status('audit_review_required',research_complete=False,next_action='Review both the development scorecard and zero-update numerical audit')


if __name__=='__main__':main()
