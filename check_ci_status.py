#!/usr/bin/env python3
"""Check GitHub Actions CI status for the latest commit."""

import json
import sys
import time
import urllib.request
from datetime import datetime

REPO = "just-do-halee/stsw"
WORKFLOW_NAMES = ["CI", "Release"]

def get_latest_commit():
    """Get the latest commit SHA."""
    url = f"https://api.github.com/repos/{REPO}/commits/main"
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read())
            return data['sha'][:7], data['commit']['message'].split('\n')[0]
    except Exception as e:
        print(f"Error getting latest commit: {e}")
        return None, None

def check_workflow_runs():
    """Check workflow runs for the latest commit."""
    sha, message = get_latest_commit()
    if not sha:
        return
    
    print(f"Latest commit: {sha} - {message}")
    print(f"Checking workflows...\n")
    
    for workflow_name in WORKFLOW_NAMES:
        url = f"https://api.github.com/repos/{REPO}/actions/workflows"
        try:
            with urllib.request.urlopen(url) as response:
                workflows = json.loads(response.read())
                
                # Find workflow by name
                workflow_id = None
                for wf in workflows['workflows']:
                    if wf['name'] == workflow_name:
                        workflow_id = wf['id']
                        break
                
                if not workflow_id:
                    print(f"❓ {workflow_name}: Workflow not found")
                    continue
                
                # Get latest run
                runs_url = f"https://api.github.com/repos/{REPO}/actions/workflows/{workflow_id}/runs?per_page=1"
                with urllib.request.urlopen(runs_url) as response:
                    runs = json.loads(response.read())
                    
                    if not runs['workflow_runs']:
                        print(f"⏸️  {workflow_name}: No runs yet")
                        continue
                    
                    run = runs['workflow_runs'][0]
                    status = run['status']
                    conclusion = run['conclusion']
                    
                    if status == 'completed':
                        if conclusion == 'success':
                            print(f"✅ {workflow_name}: {conclusion}")
                        elif conclusion == 'failure':
                            print(f"❌ {workflow_name}: {conclusion}")
                        else:
                            print(f"⚠️  {workflow_name}: {conclusion}")
                    else:
                        print(f"🔄 {workflow_name}: {status}")
                        
        except Exception as e:
            print(f"❌ {workflow_name}: Error checking - {e}")
    
    print(f"\nView details at: https://github.com/{REPO}/actions")

if __name__ == "__main__":
    check_workflow_runs()