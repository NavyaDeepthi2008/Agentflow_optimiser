"""Stage 7: evidence-based learning of conservative reuse rules.

``miner`` proposes rules from stored training runs, ``validation`` checks a
rule on held-out tasks (rule off vs on). Activation is a human decision made
through the learning service; the gateway re-checks every rule against the
tool's fixed limits on every request (``app.optimizer.learned``).
"""
