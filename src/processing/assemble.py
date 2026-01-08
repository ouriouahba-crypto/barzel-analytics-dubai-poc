from .facts import market_facts
from .descriptors import descriptors
from .proxies import proxies
from .quality import quality

def assemble(df):
    f = market_facts(df)
    return {
        "facts": f,
        "descriptors": descriptors(f),
        "proxies": proxies(f),
        "quality": quality(f),
    }
