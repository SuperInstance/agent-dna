# agent-dna

A Cloudflare Worker that serves a landing page for the Agent DNA module — a system for encoding vessel capability genomes, behavior traits, inheritance patterns, and mutation tracking across the Cocapn fleet.

## What It Does

- Serves a static HTML dashboard at `/` describing the agent DNA concept
- Provides a `/health` endpoint returning `{"status":"ok","timestamp":"..."}`

This is a presentational/documentation service. The actual DNA logic is described in the dashboard content: capability genomes, behavior traits, inheritance patterns, and mutation tracking for 200+ autonomous fleet vessels.

## Endpoints

| Path | Method | Description |
|------|--------|-------------|
| `/` | GET | HTML dashboard |
| `/health` | GET | JSON health check |

## Deploy

```bash
npx wrangler deploy
```

## License

MIT

---

Part of the [Cocapn fleet](https://github.com/Lucineer/the-fleet). Built with [Cocapn](https://github.com/Lucineer/cocapn-ai).
