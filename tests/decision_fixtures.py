"""Explicit review binding for synthetic fixtures; never used by production."""
import research_agent as agent


def bind_decisions(request):
    if request.decisions is None:
        return request
    result = agent.run_audit_only(request)
    current = (result.decision_contract or {}).get('current_review_binding')
    if current:
        request.decisions['review_binding'] = current
        request.decisions.setdefault('dataset_sha256', current['dataset_sha256'])
    return request
