"""离线存储演示，不调用 Agent、不生成文章、不执行外部写操作。"""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from media_operations.persistence.repository import MediaRepository
from media_operations.schemas import AccountCreate, AgentTaskResult, Payload, RunCreate, StrategyCreate


def demonstrate(database):
    sample_path = Path(__file__).resolve().parents[1] / "data" / "samples" / "media_account.json"
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    repository = MediaRepository(database)
    account = repository.create_account(
        AccountCreate.model_validate(sample["account"]), StrategyCreate.model_validate(sample["strategy"]),
    )
    request = RunCreate(idempotency_key="storage-demo", goal="离线验证六个阶段的存储，不生成真实内容")
    run = repository.create_run(account.account_id, request)
    repeated = repository.create_run(account.account_id, request)
    repository.start_run(account.account_id, run.run_id)
    for task in repository.list_tasks(account.account_id, run.run_id):
        repository.start_task(account.account_id, run.run_id, task.task_id)
        repository.complete_task(account.account_id, run.run_id, task.task_id, AgentTaskResult[Payload](
            success=True, data={"simulation": True, "stage": task.task_type.value, "note": "仅验证阶段结果存储"},
        ))
    repository.finish_run(account.account_id, run.run_id)
    # 新建 Repository 验证没有依赖进程内状态。
    reopened = MediaRepository(database)
    restored = reopened.get_run(account.account_id, run.run_id)
    print(json.dumps({
        "simulation": True, "account_id": account.account_id, "run_id": run.run_id,
        "duplicate_returns_same_run": run.run_id == repeated.run_id,
        "restored_status": restored.status.value,
        "tasks": [{"stage": task.task_type.value, "status": task.status.value}
                  for task in reopened.list_tasks(account.account_id, run.run_id)],
        "events": len(reopened.list_events(account.account_id, run.run_id)),
    }, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, help="显式保存到此业务库；默认使用自动清理的临时库")
    args = parser.parse_args()
    if args.database:
        demonstrate(args.database)
    else:
        with TemporaryDirectory(prefix="media-storage-demo-") as directory:
            demonstrate(Path(directory) / "media.sqlite3")


if __name__ == "__main__":
    main()
