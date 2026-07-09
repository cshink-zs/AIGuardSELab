from typing import Any

import requests
from dataclasses import dataclass
#from langchain.agents.middleware import AgentMiddleware, AgentState
#from langchain.agents.middleware.types import StateT
#from langgraph.runtime import Runtime
#from langgraph.typing import ContextT


@dataclass
class GuardResult:
    transaction_id: str
    action: str
    severity: Optional[str]
    direction: Optional[str]
    masked_content: Optional[str]
    detector_responses: Dict[str, Any]
    raw: Dict[str, Any]

    @property
    def is_blocked(self) -> bool:
        return (self.action or "").upper() == "BLOCK"

    def triggered_detectors(self) -> Dict[str, Dict[str, Any]]:
        dets = self.detector_responses or {}
        return {name: det for name, det in dets.items() if det.get("triggered") is True}

class AIGuardClient:
    def __init__(self, bearer_token: str, policy_id: str, timeout_s: int = 15):
        self.url = "https://api.zseclipse.net/v1/detection/execute-policy"
        self.bearer_token = bearer_token
        self.policy_id = str(policy_id)
        self.timeout_s = timeout_s

    def evaluate(self, *, direction: str, content: str) -> GuardResult:
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "policyId": self.policy_id,
            "direction": direction,  # "IN" or "OUT"
            "content": content,
        }

        r = requests.post(self.url, json=payload, headers=headers, timeout=self.timeout_s,verify=False)
        r.raise_for_status()
        data = r.json()

        return GuardResult(
            transaction_id=data.get("transactionId", ""),
            action=data.get("action", "ALLOW"),
            severity=data.get("severity"),
            direction=data.get("direction"),
            masked_content=data.get("maskedContent"),
            detector_responses=data.get("detectorResponses") or {},
            raw=data,
        )

    def enforce(self, *, direction: str, content: str) -> Tuple[str, GuardResult]:
        res = self.evaluate(direction=direction, content=content)

        if res.is_blocked:
            # You can tailor the message based on severity and detectors.
            raise ValueError(
                f"Blocked by Zscaler AI Guard (transactionId={res.transaction_id}, severity={res.severity})"
            )

        safe_content = res.masked_content if res.masked_content else content
        return safe_content, res


#class AIGuardMiddleware(AgentMiddleware):

 #   def before_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
 #       return None

  #  def after_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
   #     return None