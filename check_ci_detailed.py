#!/usr/bin/env python3
"""Check detailed GitHub Actions CI status."""

import json
import urllib.request

REPO = "just-do-halee/stsw"


def get_latest_run():
    """Get the latest CI workflow run."""
    url = f"https://api.github.com/repos/{REPO}/actions/runs?per_page=1"
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read())
            if data["workflow_runs"]:
                return data["workflow_runs"][0]
    except Exception as e:
        print(f"Error: {e}")
    return None


def get_jobs(run_id: int):
    """Get jobs for a run."""
    url = f"https://api.github.com/repos/{REPO}/actions/runs/{run_id}/jobs"
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read())
            return data["jobs"]
    except Exception as e:
        print(f"Error getting jobs: {e}")
    return []


def main():
    run = get_latest_run()
    if not run:
        print("No workflow runs found")
        return

    print(f"Workflow: {run['name']}")
    print(f"Status: {run['status']}")
    print(f"URL: {run['html_url']}")
    print("\nJobs:")

    jobs = get_jobs(run["id"])

    # Group by status
    completed = []
    running = []
    queued = []

    for job in jobs:
        if job["status"] == "completed":
            completed.append(job)
        elif job["status"] == "in_progress":
            running.append(job)
        else:
            queued.append(job)

    # Show running jobs first
    if running:
        print("\n🔄 Running:")
        for job in running:
            print(f"  - {job['name']}")

    # Show completed jobs
    if completed:
        print(f"\n✅ Completed ({len(completed)}):")
        success = [j for j in completed if j["conclusion"] == "success"]
        failed = [j for j in completed if j["conclusion"] != "success"]

        if success:
            print(f"  ✓ {len(success)} passed")
        if failed:
            print(f"  ✗ {len(failed)} failed:")
            for job in failed:
                print(f"    - {job['name']}: {job['conclusion']}")

    # Show queued
    if queued:
        print(f"\n⏳ Queued: {len(queued)}")

    print(f"\nTotal: {len(jobs)} jobs")


if __name__ == "__main__":
    main()
