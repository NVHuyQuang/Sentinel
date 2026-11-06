import sys
import json
import time

# Bypass torchaudio DLL loading issue on Windows if present
sys.modules['torchaudio'] = None

from laya import Router

def main():
    print("Loading Laya model on CPU...")
    # Explicitly force CPU to prevent GPU/CUDA driver timeouts
    router = Router(device="cpu")
    
    text = (
        "I was excited to launch the new dashboard because the early results looked "
        "promising, but the release became stressful when the API started timing out "
        "during peak traffic. The team stayed late to investigate, and we eventually "
        "found a caching bug that caused stale customer data to appear for roughly "
        "twenty minutes. No permanent data was lost, and the issue has been fixed, "
        "but I am disappointed that we did not catch it before deployment. I appreciate "
        "everyone who helped recover the service, although I would not call this release "
        "a complete success yet because we still need stronger load tests and monitoring."
    )
    questions = {
        "sentiment": {
            "type": "choice",
            "instructions": (
                "What is the overall sentiment of the author? Weigh the final assessment "
                "more heavily than isolated positive or negative phrases."
            ),
            "criteria": {
                "positive": "enthusiastic, grateful, optimistic, satisfied",
                "negative": "angry, hopeless, hostile, deeply dissatisfied",
                "mixed": "contains both appreciation or optimism and disappointment or concern",
                "neutral": "strictly factual with no meaningful emotional evaluation",
            },
        },
        "incident_type": {
            "type": "choice",
            "instructions": "What best describes the main issue reported in the text?",
            "criteria": {
                "performance": "timeouts, slowness, overload, or degraded service performance",
                "data_integrity": "permanent data loss, corruption, unauthorized changes, or incorrect records",
                "security": "authentication, authorization, privacy, breach, or malicious activity",
                "usability": "confusing interface, difficult workflow, or user experience problem",
                "no_incident": "no concrete operational issue is described",
            },
        },
        "urgency": {
            "type": "choice",
            "instructions": (
                "How urgent is the situation now, considering both the historical incident "
                "and the current state at the end of the text?"
            ),
            "criteria": {
                "critical": "active severe outage, ongoing data loss, or immediate danger",
                "high": "serious unresolved impact requiring immediate response",
                "medium": "the incident is contained, but follow-up work is important",
                "low": "minor issue with little or no meaningful impact",
            },
        }
    }
    
    # Warmup / load model into memory
    router.predict("warmup", questions)
    
    print(f"Input: {text}")
    print("Running CPU inference...")
    
    t0 = time.perf_counter()
    result = router.predict(text, questions)
    latency = (time.perf_counter() - t0) * 1000
    
    print(f"\nInference completed in {latency:.2f} ms")
    print("\nResult:")
    print(json.dumps(result["answers"]["sentiment"], indent=2))

if __name__ == "__main__":
    main()
