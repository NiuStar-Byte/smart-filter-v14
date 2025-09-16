# check_symbols.py

import ccxt

# Your current TOKENS list

TOKENS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "ADA-USDT",
    "AVAX-USDT", "XLM-USDT", "LINK-USDT", "POL-USDT", "BNB-USDT",
    "SKATE-USDT", "LA-USDT", "SPK-USDT", "ZKJ-USDT", "IP-USDT",
    "AERO-USDT", "BMT-USDT", "LQTY-USDT", "X-USDT", "RAY-USDT",
    "EPT-USDT", "ELDE-USDT", "MAGIC-USDT", "ACTSOL-USDT", "FUN-USDT",
    "CROSS-USDT", "KNC-USDT", "AIN-USDT", "ARK-USDT", "PORTAL-USDT",
    "ICNT-USDT", "OMNI-USDT", "PARTI-USDT", "VINE-USDT", "ZORA-USDT",
    "DUCK-USDT", "AUCTION-USDT", "ROAM-USDT", "FUEL-USDT", "TUT-USDT",
    "VOXEL-USDT", "ALU-USDT", "TURBO-USDT", "PROMPT-USDT", "HIPPO-USDT", 
    "DOGE-USDT", "ALGO-USDT", "DOT-USDT", "NEWT-USDT", "SAHARA-USDT",
    "PEPE-USDT", "ERA-USDT", "PENGU-USDT", "CFX-USDT", "ENA-USDT",
    "SUI-USDT", "EIGEN-USDT", "UNI-USDT", "HYPE-USDT", "TON-USDT",
    "KAS-USDT", "HBAR-USDT", "ONDO-USDT", "VIRTUAL-USDT", "AAVE-USDT",
    "GALA-USDT", "PUMP-USDT", "PEPE-USDT", "WIF-USDT", "BERA-USDT", "DYDX-USDT",
    "KAITO-USDT", "ARKM-USDT", "ATH-USDT", "NMR-USDT", "ARB-USDT",
    "WLFI-USDT"
 ]

# Load market symbols from each exchange
ku = ccxt.kucoin()
bn = ccxt.binance()
ku_markets = set(ku.load_markets().keys())
bn_markets = set(bn.load_markets().keys())

# Check each token
for tok in TOKENS:
    hyphen = tok
    slash  = tok.replace('-', '/')
    nobar  = tok.replace('-', '')
    ok_ku  = (slash in ku_markets) or (hyphen in ku_markets)
    ok_bn  = nobar in bn_markets
    print(f"{tok:12} → KuCoin? {ok_ku}  Binance? {ok_bn}")

TOKEN_BLOCKCHAIN_INFO = {
    # Top Layer-1s and well-known tokens
    "BTC-USDT":   {"base": "BTC", "blockchain": "Bitcoin",      "consensus": "PoW"},
    "ETH-USDT":   {"base": "ETH", "blockchain": "Ethereum",     "consensus": "PoS"},
    "SOL-USDT":   {"base": "SOL", "blockchain": "Solana",       "consensus": "PoH+PoS"},
    "XRP-USDT":   {"base": "XRP", "blockchain": "XRPLedger",    "consensus": "Ripple"},
    "ADA-USDT":   {"base": "ADA", "blockchain": "Cardano",      "consensus": "Ouroboros PoS"},
    "AVAX-USDT":  {"base": "AVAX","blockchain": "Avalanche",    "consensus": "Avalanche"},
    "XLM-USDT":   {"base": "XLM", "blockchain": "Stellar",      "consensus": "SCP/FBA"},
    "LINK-USDT":  {"base": "LINK","blockchain": "Chainlink",    "consensus": "Hybrid PoS"},
    "POL-USDT":   {"base": "POL", "blockchain": "Polygon",      "consensus": "PoS"},
    "BNB-USDT":   {"base": "BNB", "blockchain": "BSC",          "consensus": "PoSA"},
    "DOGE-USDT":  {"base": "DOGE","blockchain": "Dogecoin",     "consensus": "PoW"},
    "ALGO-USDT":  {"base": "ALGO","blockchain": "Algorand",     "consensus": "Pure PoS"},
    "DOT-USDT":   {"base": "DOT", "blockchain": "Polkadot",     "consensus": "NPoS"},
    "TON-USDT":   {"base": "TON", "blockchain": "TON",          "consensus": "PoS"},
    "KAS-USDT":   {"base": "KAS", "blockchain": "Kaspa",        "consensus": "PoW"},
    "HBAR-USDT":  {"base": "HBAR","blockchain": "Hedera",       "consensus": "Hashgraph"},
    "SUI-USDT":   {"base": "SUI", "blockchain": "Sui",          "consensus": "DPoS"},
    "CFX-USDT":   {"base": "CFX", "blockchain": "Conflux",      "consensus": "Tree-Graph"},
    "PEPE-USDT":  {"base": "PEPE","blockchain": "Ethereum",     "consensus": "PoS"},
    "ERA-USDT":   {"base": "ERA", "blockchain": "zkSync",       "consensus": "PoS"},
    "WIF-USDT":   {"base": "WIF", "blockchain": "Solana",       "consensus": "PoH+PoS"},
    "UNI-USDT":   {"base": "UNI", "blockchain": "Ethereum",     "consensus": "PoS"},
    "AAVE-USDT":  {"base": "AAVE","blockchain": "Ethereum",     "consensus": "PoS"},
    "GALA-USDT":  {"base": "GALA","blockchain": "Ethereum",     "consensus": "PoS"},
    "ONDO-USDT":  {"base": "ONDO","blockchain": "Ethereum",     "consensus": "PoS"},
    "VIRTUAL-USDT":{"base": "VIRTUAL", "blockchain": "Ethereum", "consensus": "PoS"},

    # Other major Layer-1s and Layer-2s, DeFi, NFT, meme, and ecosystem tokens
    "SKATE-USDT":   {"base": "SKATE",   "blockchain": "Ethereum",     "consensus": "PoS"},
    "LA-USDT":      {"base": "LA",      "blockchain": "Ethereum",     "consensus": "PoS"},
    "SPK-USDT":     {"base": "SPK",     "blockchain": "Solana",       "consensus": "PoH+PoS"},
    "ZKJ-USDT":     {"base": "ZKJ",     "blockchain": "zkSync",       "consensus": "PoS"},
    "IP-USDT":      {"base": "IP",      "blockchain": "Ethereum",     "consensus": "PoS"},
    "AERO-USDT":    {"base": "AERO",    "blockchain": "Ethereum",     "consensus": "PoS"},
    "BMT-USDT":     {"base": "BMT",     "blockchain": "Ethereum",     "consensus": "PoS"},
    "LQTY-USDT":    {"base": "LQTY",    "blockchain": "Ethereum",     "consensus": "PoS"},
    "X-USDT":       {"base": "X",       "blockchain": "Ethereum",     "consensus": "PoS"},
    "RAY-USDT":     {"base": "RAY",     "blockchain": "Solana",       "consensus": "PoH+PoS"},
    "EPT-USDT":     {"base": "EPT",     "blockchain": "Ethereum",     "consensus": "PoS"},
    "ELDE-USDT":    {"base": "ELDE",    "blockchain": "Ethereum",     "consensus": "PoS"},
    "MAGIC-USDT":   {"base": "MAGIC",   "blockchain": "Arbitrum",     "consensus": "PoS"},
    "ACTSOL-USDT":  {"base": "ACTSOL",  "blockchain": "Solana",       "consensus": "PoH+PoS"},
    "FUN-USDT":     {"base": "FUN",     "blockchain": "Ethereum",     "consensus": "PoS"},
    "CROSS-USDT":   {"base": "CROSS",   "blockchain": "Ethereum",     "consensus": "PoS"},
    "KNC-USDT":     {"base": "KNC",     "blockchain": "Ethereum",     "consensus": "PoS"},
    "AIN-USDT":     {"base": "AIN",     "blockchain": "Ethereum",     "consensus": "PoS"},
    "ARK-USDT":     {"base": "ARK",     "blockchain": "ARK",          "consensus": "DPoS"},
    "PORTAL-USDT":  {"base": "PORTAL",  "blockchain": "Ethereum",     "consensus": "PoS"},
    "ICNT-USDT":    {"base": "ICNT",    "blockchain": "Ethereum",     "consensus": "PoS"},
    "OMNI-USDT":    {"base": "OMNI",    "blockchain": "Omni",         "consensus": "PoW"},
    "PARTI-USDT":   {"base": "PARTI",   "blockchain": "Ethereum",     "consensus": "PoS"},
    "VINE-USDT":    {"base": "VINE",    "blockchain": "Ethereum",     "consensus": "PoS"},
    "ZORA-USDT":    {"base": "ZORA",    "blockchain": "Ethereum",     "consensus": "PoS"},
    "DUCK-USDT":    {"base": "DUCK",    "blockchain": "Ethereum",     "consensus": "PoS"},
    "AUCTION-USDT": {"base": "AUCTION", "blockchain": "Ethereum",     "consensus": "PoS"},
    "ROAM-USDT":    {"base": "ROAM",    "blockchain": "Ethereum",     "consensus": "PoS"},
    "FUEL-USDT":    {"base": "FUEL",    "blockchain": "Ethereum",     "consensus": "PoS"},
    "TUT-USDT":     {"base": "TUT",     "blockchain": "Ethereum",     "consensus": "PoS"},
    "VOXEL-USDT":   {"base": "VOXEL",   "blockchain": "Ethereum",     "consensus": "PoS"},
    "ALU-USDT":     {"base": "ALU",     "blockchain": "BNB Chain",    "consensus": "PoSA"},
    "TURBO-USDT":   {"base": "TURBO",   "blockchain": "Ethereum",     "consensus": "PoS"},
    "PROMPT-USDT":  {"base": "PROMPT",  "blockchain": "Ethereum",     "consensus": "PoS"},
    "HIPPO-USDT":   {"base": "HIPPO",   "blockchain": "Ethereum",     "consensus": "PoS"},
    "NEWT-USDT":    {"base": "NEWT",    "blockchain": "Ethereum",     "consensus": "PoS"},
    "SAHARA-USDT":  {"base": "SAHARA",  "blockchain": "Ethereum",     "consensus": "PoS"},
    "PENGU-USDT":   {"base": "PENGU",   "blockchain": "Solana",       "consensus": "PoH+PoS"},
    "ENA-USDT":     {"base": "ENA",     "blockchain": "Ethereum",     "consensus": "PoS"},
    "EIGEN-USDT":   {"base": "EIGEN",   "blockchain": "Ethereum",     "consensus": "PoS"},
    "HYPE-USDT":    {"base": "HYPE",    "blockchain": "Solana",       "consensus": "PoH+PoS"},
    "PUMP-USDT":    {"base": "PUMP",    "blockchain": "Ethereum",     "consensus": "PoS"},
    "BERA-USDT":    {"base": "BERA",    "blockchain": "Berachain",    "consensus": "PoL"},
    "DYDX-USDT":    {"base": "DYDX",    "blockchain": "DYDX Chain",   "consensus": "PoS Cosmos"}

    # Add further tokens below as needed, with best-guess or "Unknown" if truly uncertain:
    # Examples:
    # "TOKEN-USDT": {"base": "TOKEN", "blockchain": "Ethereum", "consensus": "PoS"},
}

def get_token_blockchain_info(symbol: str):
    """
    Returns blockchain and consensus info for a given symbol.
    If not found, returns None.
    """
    return TOKEN_BLOCKCHAIN_INFO.get(symbol, None)

def all_supported_tokens():
    """
    Returns the full list of supported tokens.
    """
    return list(TOKEN_BLOCKCHAIN_INFO.keys())
