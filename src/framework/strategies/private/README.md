# Private strategies

Drop real strategy modules in this directory. Everything here **except this
README and `__init__.py` is gitignored**, so proprietary strategy logic never
reaches the public repository — only the framework and the example strategy
are published.

A strategy module looks like:

```python
import pandas as pd
from pydantic import Field

from framework.strategies import register
from framework.strategies.base import Strategy, StrategyParams


@register
class MyStrategy(Strategy):
    name = "my_strategy"

    class Params(StrategyParams):
        lookback: int = Field(60, ge=2)

    params_model = Params

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        ...  # return target weights in [0, 1], one column per symbol
```

Pair it with a config in `config/strategies/private/` (also gitignored):

```yaml
strategy: my_strategy
symbols: [SPY]
params:
  lookback: 60
```

Modules here are auto-imported, so the strategy is immediately visible to
`tf list-strategies` and runnable with `tf run`.
