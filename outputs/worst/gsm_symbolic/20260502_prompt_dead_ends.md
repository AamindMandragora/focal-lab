Task family: gsm_symbolic.
Recent prompt-level dead ends to avoid before writing a new strategy.
Do not use explicit delimiter helpers (`AppendLeftDelimiter`, `AppendRightDelimiter`, `AppendForcedToken`, `ForcedTokenStep`) in GSM natural-delimiter runs.
Do not switch into `final_ready`, `answer_ready`, `phase = "open"`, or similar span-opening state in a branch that consumes no helper step; the no-progress guard will terminate the loop.
Do not use tiny fixed quotas such as 4, 6, 8, 10, or 12 reasoning/wrap/answer steps as the main finality signal.
Do not set final-answer readiness from an early counter like `reason_steps >= 24` or after only a few raw cues; that opens on the first local arithmetic fragment.
Preferred directions are either a durable late-open final span after forty-plus reasoning/setup steps, or a scratch-to-final policy where raw unconstrained steps observe `=` cues and later open reusable arithmetic spans.
