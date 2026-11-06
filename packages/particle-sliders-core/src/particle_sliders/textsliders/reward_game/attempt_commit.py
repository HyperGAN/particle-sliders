"""An attempt is terminal only after its evidence archive is complete."""
from .core import IntegrityError, digest, immutable, read, write


def commit_decision(folder, decision, archive):
    # Archival is idempotent and may fail after writing a retained prefix.
    # Never let that prefix hide the attempt from the recovery search.
    archive()
    immutable(folder / 'archive-complete.json', {'decision_sha256': digest(decision)})
    immutable(folder / 'decision.json', decision)
    write(folder / 'status.json', dict(state='decided', **decision))


def decision_committed(folder):
    decision = folder / 'decision.json'
    receipt = folder / 'archive-complete.json'
    if not decision.exists() or not receipt.exists():
        return False
    if read(receipt) != {'decision_sha256': digest(read(decision))}:
        raise IntegrityError('Attempt archive receipt differs from decision')
    return True
