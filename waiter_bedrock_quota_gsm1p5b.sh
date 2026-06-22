#!/bin/bash
# Waiter: polls Bedrock quota every 30 minutes.
# When the quota resets (probe call succeeds), execs the GSM-1.5B seed123 launcher.
set -uo pipefail
cd /home/aadivyar/csd-generation

# Load .env credentials (AWS_BEARER_TOKEN_BEDROCK, AWS_REGION, etc.)
set -a; source /home/aadivyar/csd-generation/.env; set +a

export LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-}

LAUNCHER=/home/aadivyar/csd-generation/launch_gsm1p5b_seed123_fresh_20260611.sh
PYTHON=/apps/conda/advayth2/envs/advayth2/bin/python
MODEL=us.anthropic.claude-sonnet-4-6
REGION=${AWS_REGION:-us-east-1}
LOG=/home/aadivyar/csd-generation/waiter_bedrock_quota_gsm1p5b.log

echo "WAITER_START $(date)" | tee -a "$LOG"
echo "PROBE_TARGET model=$MODEL region=$REGION" | tee -a "$LOG"

probe_bedrock() {
  # Returns 0 if Bedrock responds (quota available), 1 if still throttled/blocked.
  "$PYTHON" - <<'PYEOF'
import os, sys, urllib.request, urllib.error, json

token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "")
region = os.environ.get("AWS_REGION", "us-east-1")
model = "us.anthropic.claude-sonnet-4-6"
base_url = os.environ.get("BEDROCK_BASE_URL") or f"https://bedrock-runtime.{region}.amazonaws.com"
import urllib.parse
model_enc = urllib.parse.quote(model, safe="")
url = f"{base_url}/model/{model_enc}/converse"

payload = json.dumps({
    "system": [{"text": "You are a test probe."}],
    "messages": [{"role": "user", "content": [{"text": "hi"}]}],
    "inferenceConfig": {"maxTokens": 1},
}).encode("utf-8")

req = urllib.request.Request(
    url,
    data=payload,
    headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    },
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read()
        sys.exit(0)  # success — quota available
except urllib.error.HTTPError as e:
    body = e.read().decode("utf-8", errors="replace")
    if e.code == 429 or "Too many tokens" in body or "throttl" in body.lower() or "quota" in body.lower():
        sys.exit(2)  # quota still blocked
    # Any other HTTP error: treat as transient
    sys.stderr.write(f"HTTP_ERROR {e.code}: {body[:200]}\n")
    sys.exit(3)
except Exception as ex:
    sys.stderr.write(f"PROBE_EXCEPTION: {ex}\n")
    sys.exit(3)
PYEOF
}

while true; do
  PROBE_OUT=$( probe_bedrock 2>&1 )
  PROBE_RC=$?

  if [ "$PROBE_RC" -eq 0 ]; then
    echo "QUOTA_RESET_DETECTED $(date)" | tee -a "$LOG"
    echo "LAUNCHING: $LAUNCHER" | tee -a "$LOG"
    exec bash -l "$LAUNCHER" >> "$LOG" 2>&1

  elif [ "$PROBE_RC" -eq 2 ]; then
    echo "QUOTA_STILL_BLOCKED $(date)" | tee -a "$LOG"

  else
    # Unexpected error — log it and keep retrying
    echo "PROBE_ERROR rc=$PROBE_RC $(date): $PROBE_OUT" | tee -a "$LOG"
  fi

  sleep 1800  # 30 minutes
done
