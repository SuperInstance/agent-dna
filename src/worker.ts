interface GeneticTrait {
  id: string;
  name: string;
  value: number;
  min: number;
  max: number;
  dominant: boolean;
  inheritedFrom?: string;
  mutationRate: number;
}

interface CapabilityGene {
  code: string;
  category: 'perception' | 'action' | 'cognition' | 'communication';
  level: number;
  active: boolean;
  dependencies: string[];
}

interface AgentDNA {
  agentId: string;
  generation: number;
  lineage: string[];
  traits: GeneticTrait[];
  capabilities: CapabilityGene[];
  mutationHistory: MutationRecord[];
  fitnessScore: number;
  diversityIndex: number;
  timestamp: number;
}

interface MutationRecord {
  id: string;
  timestamp: number;
  traitId: string;
  previousValue: number;
  newValue: number;
  mutationType: 'point' | 'insertion' | 'deletion' | 'crossover';
  environmentalFactor?: string;
}

interface PopulationMetrics {
  totalAgents: number;
  averageFitness: number;
  geneticDiversity: number;
  mutationRate: number;
  dominantTraits: string[];
  extinctLineages: string[];
}

const DNA_STORAGE: Record<string, AgentDNA> = {};
const TRAIT_REGISTRY: GeneticTrait[] = [];
const POPULATION_METRICS: PopulationMetrics = {
  totalAgents: 0,
  averageFitness: 0,
  geneticDiversity: 1.0,
  mutationRate: 0.01,
  dominantTraits: [],
  extinctLineages: []
};

function initializeDefaultTraits(): void {
  const defaultTraits: Omit<GeneticTrait, 'id'>[] = [
    { name: 'aggression', value: 0.5, min: 0, max: 1, dominant: true, mutationRate: 0.05 },
    { name: 'curiosity', value: 0.7, min: 0, max: 1, dominant: false, mutationRate: 0.03 },
    { name: 'cooperation', value: 0.6, min: 0, max: 1, dominant: true, mutationRate: 0.04 },
    { name: 'adaptability', value: 0.8, min: 0, max: 1, dominant: false, mutationRate: 0.02 },
    { name: 'persistence', value: 0.5, min: 0, max: 1, dominant: true, mutationRate: 0.06 }
  ];

  TRAIT_REGISTRY.push(...defaultTraits.map((trait, index) => ({
    ...trait,
    id: `trait_${index.toString(16).padStart(4, '0')}`
  })));
}

function generateAgentDNA(agentId: string, parentId?: string): AgentDNA {
  const parentDNA = parentId ? DNA_STORAGE[parentId] : null;
  const generation = parentDNA ? parentDNA.generation + 1 : 1;
  const lineage = parentDNA ? [...parentDNA.lineage, parentId] : [];
  
  const traits: GeneticTrait[] = TRAIT_REGISTRY.map(trait => {
    if (parentDNA) {
      const inheritedTrait = parentDNA.traits.find(t => t.name === trait.name);
      if (inheritedTrait) {
        const shouldMutate = Math.random() < trait.mutationRate;
        const newValue = shouldMutate 
          ? Math.max(trait.min, Math.min(trait.max, inheritedTrait.value + (Math.random() - 0.5) * 0.2))
          : inheritedTrait.value;
        
        return {
          ...trait,
          value: newValue,
          inheritedFrom: inheritedTrait.inheritedFrom || parentId
        };
      }
    }
    
    return {
      ...trait,
      value: trait.min + Math.random() * (trait.max - trait.min),
      inheritedFrom: 'initial'
    };
  });

  const capabilities: CapabilityGene[] = [
    { code: 'VIS_PER_01', category: 'perception', level: 1, active: true, dependencies: [] },
    { code: 'ACT_MOV_01', category: 'action', level: 1, active: true, dependencies: [] },
    { code: 'COG_DEC_01', category: 'cognition', level: 1, active: true, dependencies: [] },
    { code: 'COM_MSG_01', category: 'communication', level: 1, active: true, dependencies: [] }
  ];

  const dna: AgentDNA = {
    agentId,
    generation,
    lineage,
    traits,
    capabilities,
    mutationHistory: [],
    fitnessScore: 0.5,
    diversityIndex: calculateDiversityIndex(traits),
    timestamp: Date.now()
  };

  DNA_STORAGE[agentId] = dna;
  updatePopulationMetrics();
  return dna;
}

function calculateDiversityIndex(traits: GeneticTrait[]): number {
  if (traits.length === 0) return 0;
  const variance = traits.reduce((sum, trait) => {
    const mean = (trait.min + trait.max) / 2;
    return sum + Math.pow(trait.value - mean, 2);
  }, 0) / traits.length;
  return Math.min(1, variance * 10);
}

function updatePopulationMetrics(): void {
  const agents = Object.values(DNA_STORAGE);
  POPULATION_METRICS.totalAgents = agents.length;
  
  if (agents.length > 0) {
    POPULATION_METRICS.averageFitness = agents.reduce((sum, a) => sum + a.fitnessScore, 0) / agents.length;
    POPULATION_METRICS.geneticDiversity = agents.reduce((sum, a) => sum + a.diversityIndex, 0) / agents.length;
    
    const traitFrequency: Record<string, number> = {};
    agents.forEach(agent => {
      agent.traits.forEach(trait => {
        if (trait.dominant) {
          traitFrequency[trait.name] = (traitFrequency[trait.name] || 0) + 1;
        }
      });
    });
    
    POPULATION_METRICS.dominantTraits = Object.entries(traitFrequency)
      .filter(([_, count]) => count > agents.length * 0.7)
      .map(([name]) => name);
  }
}

function applyMutation(agentId: string, traitName: string, mutationType: MutationRecord['mutationType']): AgentDNA | null {
  const dna = DNA_STORAGE[agentId];
  if (!dna) return null;

  const traitIndex = dna.traits.findIndex(t => t.name === traitName);
  if (traitIndex === -1) return null;

  const trait = dna.traits[traitIndex];
  const previousValue = trait.value;
  let newValue = previousValue;
  
  switch (mutationType) {
    case 'point':
      newValue = Math.max(trait.min, Math.min(trait.max, previousValue + (Math.random() - 0.5) * 0.3));
      break;
    case 'insertion':
      newValue = Math.min(trait.max, previousValue + 0.1);
      break;
    case 'deletion':
      newValue = Math.max(trait.min, previousValue - 0.1);
      break;
    case 'crossover':
      const otherAgents = Object.values(DNA_STORAGE).filter(a => a.agentId !== agentId);
      if (otherAgents.length > 0) {
        const randomAgent = otherAgents[Math.floor(Math.random() * otherAgents.length)];
        const otherTrait = randomAgent.traits.find(t => t.name === traitName);
        if (otherTrait) {
          newValue = (previousValue + otherTrait.value) / 2;
        }
      }
      break;
  }

  dna.traits[traitIndex] = {
    ...trait,
    value: newValue
  };

  const mutationRecord: MutationRecord = {
    id: `mut_${Date.now().toString(16)}`,
    timestamp: Date.now(),
    traitId: trait.id,
    previousValue,
    newValue,
    mutationType,
    environmentalFactor: 'directed'
  };

  dna.mutationHistory.push(mutationRecord);
  dna.diversityIndex = calculateDiversityIndex(dna.traits);
  dna.timestamp = Date.now();
  
  updatePopulationMetrics();
  return dna;
}

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
  'Content-Security-Policy': "default-src 'self'; script-src 'self'",
  'X-Frame-Options': 'DENY',
  'X-Content-Type-Options': 'nosniff'
};

const htmlResponse = (content: string) => {
  return new Response(content, {
    headers: {
      'Content-Type': 'text/html',
      ...corsHeaders
    }
  });
};

const jsonResponse = (data: unknown, status = 200) => {
  return new Response(JSON.stringify(data, null, 2), {
    status,
    headers: {
      'Content-Type': 'application/json',
      ...corsHeaders
    }
  });
};

const errorResponse = (message: string, status = 400) => {
  return jsonResponse({ error: message }, status);
};

const healthCheck = () => {
  return jsonResponse({
    status: 'healthy',
    timestamp: Date.now(),
    agents: Object.keys(DNA_STORAGE).length,
    traits: TRAIT_REGISTRY.length,
    uptime: process.uptime ? Math.floor(process.uptime()) : 0
  });
};

const fleetFooter = `
<div style="
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  background: #0a0a0f;
  color: #e11d48;
  padding: 12px;
  font-family: monospace;
  font-size: 12px;
  border-top: 1px solid #e11d48;
  display: flex;
  justify-content: space-between;
  align-items: center;
">
  <span>agent-dna v1.0</span>
  <span>population: ${POPULATION_METRICS.totalAgents}</span>
  <span>diversity: ${POPULATION_METRICS.geneticDiversity.toFixed(3)}</span>
  <a href="/health" style="color: #e11d48; text-decoration: none;">/health</a>
</div>
`;

const landingPage = () => {
  return htmlResponse(`
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Agent DNA</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      background: #0a0a0f;
      color: #ffffff;
      font-family: 'Courier New', monospace;
      min-height: 100vh;
      padding-bottom: 50px;
    }
    .container {
      max-width: 1200px;
      margin: 0 auto;
      padding: 20px;
    }
    header {
      border-bottom: 2px solid #e11d48;
      padding-bottom: 20px;
      margin-bottom: 40px;
    }
    h1 {
      color: #e11d48;
      font-size: 3em;
      margin-bottom: 10px;
    }
    .subtitle {
      color: #888;
      font-size: 1.2em;
    }
    .endpoints {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 20px;
      margin-bottom: 40px;
    }
    .endpoint-card {
      background: rgba(225, 29, 72, 0.1);
      border: 1px solid #e11d48;
      border-radius: 8px;
      padding: 20px;
      transition: transform 0.2s;
    }
    .endpoint-card:hover {
      transform: translateY(-2px);
    }
    .method {
      display: inline-block;
      background: #e11d48;
      color: white;
      padding: 4px 12px;
      border-radius: 4px;
      font-weight: bold;
      margin-right: 10px;
    }
    .path {
      color: #e11d48;
      font-family: monospace;
      font-size: 1.1em;
    }
    .description {
      margin-top: 10px;
      color: #ccc;
    }
    .metrics {
      background: rgba(255, 255, 255, 0.05);
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 40px;
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 20px;
      margin-top: 20px;
    }
    .metric-item {
      text-align: center;
    }
    .metric-value {
      font-size: 2em;
      color: #e11d48;
      font-weight: bold;
    }
    .metric-label {
      color: #888;
      font-size: 0.9em;
      text-transform: uppercase;
      letter-spacing: 1px;
    }
    code {
      background: rgba(225, 29, 72, 0.2);
      color: #e11d48;
      padding: 2px 6px;
      border-radius: 4px;
      font-family: monospace;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Agent DNA</h1>
      <div class="subtitle">Genetic code for vessel capabilities and behavior patterns</div>
    </header>
    
    <div class="metrics">
      <h2>Population Metrics</h2>
      <div class="metric-grid">
        <div class="metric-item">
          <div class="metric-value">${POPULATION_METRICS.totalAgents}</div>
          <div class="metric-label">Active Agents</div>
        </div>
        <div class="metric-item">
          <div class="metric-value">${POPULATION_METRICS.averageFitness.toFixed(3)}</div>
          <div class="metric-label">Avg Fitness</div>
        </div>
        <div class="metric-item">
          <div class="metric-value">${POPULATION_METRICS.geneticDiversity.toFixed(3)}</div>
          <div class="metric-label">Diversity Index</div>
        </div>
        <div class="metric-item">
          <div class="metric-value">${POPULATION_METRICS.mutationRate.toFixed(3)}</div>
          <div class="metric-label">Mutation Rate</div>
        </div>
      </div>
    </div>
    
    <div class="endpoints">
      <div class="endpoint-card">
        <div><span class="method">GET</span> <span class="path">/api/dna/:agent</span></div>
        <div class="description">Retrieve genetic code for specific agent</div>
      </div>
      <div class="endpoint-card">
        <div><span class="method">GET</span> <span class="path">/api/traits</span></div>
        <div class="description">Get all registered behavior traits</div>
      </div>
      <div class="endpoint-card">
        <div><span class="method">POST</span> <span class="path">/api/mutate</span></div>
        <div class="description">Apply directed mutation to agent DNA</div>
      </div>
      <div class="endpoint-card">
        <div><span class="method">GET</span> <span class="path">/api/population</span></div>
        <div class="description">Get population genetic metrics</div>
      </div>
      <div class="endpoint-card">
        <div><span class="method">POST</span> <span class="path">/api/agent</span></div>
        <div class="description">Create new agent (optionally with parent)</div>
      </div>
      <div class="endpoint-card">
        <div><span class="method">GET</span> <span class="path">/health</span></div>
        <div class="description">System health check</div>
      </div>
    </div>
    
    <div style="color: #666; font-size: 0.9em; margin-top: 40px; padding-top: 20px; border-top: 1px solid #333;">
      <p>Use <code>curl -X GET https://agent-dna.workers.dev/api/traits</code> to test the API</p>
      <p>All responses include CSP and X-Frame-Options headers</p>
    </div>
  </div>
  
  ${fleetFooter}
</body>
</html>
  `);
};

async function handleRequest(request: Request): Promise<Response> {
  const url = new URL(request.url);
  const path = url.pathname;
  
  if (request.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }
  
  if (path === '/' || path === '') {
    return landingPage();
  }
  
  if (path === '/health') {
    return healthCheck();
  }
  
  if (path.startsWith('/api/')) {
    if (path.startsWith('/api/dna/')) {
      const agentId = path.split('/api/dna/')[1];
      if (!agentId) {
        return errorResponse('Agent ID required');
      }
      
      if (request.method !== 'GET') {
        return errorResponse('Method not allowed', 405);
      }
      
      const dna = DNA_STORAGE[agentId];
      if (!dna) {
        return errorResponse('Agent not found', 404);
      }
      
      return jsonResponse(dna);
    }
    
    if (path === '/api/traits') {
      if (request.method !== 'GET') {
        return errorResponse('Method not allowed', 405);
      }
      return jsonResponse(TRAIT_REGISTRY);
    }
    
    if (path === '/api/population') {
      if (request.method !== 'GET') {
        return errorResponse('Method not allowed', 405);
      }
      return jsonResponse(POPULATION_METRICS);
    }
    
    if (path === '/api/mutate') {
      if (request.method !== 'POST') {
        return errorResponse('Method not allowed', 405);
      }
      
      try {
        const body = await request.json() as { agentId: string; traitName: string; mutationType?: MutationRecord['mutationType'] };
        
        if (!body.agentId || !body.traitName) {
          return errorResponse('agentId and traitName required');
        }
        
        const mutationType = body.mutationType || 'point';
        const result = applyMutation(body.agentId, body.traitName, mutationType);
        
        if (!result) {
          return errorResponse('Mutation failed', 404);
        }
        
        return jsonResponse({
          success: true,
          agentId: body.agentId,
          traitName: body.traitName,
          mutationType,
          newValue: result.traits.find(t => t.name === body.traitName)?.value,
          diversityIndex: result.diversityIndex
        });
      } catch {
        return errorResponse('Invalid JSON body');
      }
    }
    
    if (path === '/api/agent') {
      if (request.method === 'GET') {
        return jsonResponse(Object.keys(DNA_STORAGE));
      }
      
      if (request.method === 'POST') {
        try {
          const body = await request.json() as { agentId: string; parentId?: string };
          
          if (!body.agentId) {
            return errorResponse('agentId required');
          }
          
          if (DNA_STORAGE[body.agentId]) {
            return errorResponse('Agent already exists', 409);
          }
          
          const dna = generateAgentDNA(body.agentId, body.parentId);
          return jsonResponse({
            success: true,
            agentId: body.agentId,
            generation: dna.generation,
            lineage: dna.lineage,
            traits: dna.traits.length,
            capabilities: dna.capabilities.length
          }, 201);
        } catch {
          return errorResponse('Invalid JSON body');
        }
      }
      
      return errorResponse('Method not allowed', 405);
    }
    
    return errorResponse('Endpoint not found', 404);
  }
  
  return errorResponse('Not found', 404);
}

initializeDefaultTraits();

export default {
  async fetch(request: Request): Promise<Response> {
    return handleRequest(request);
  }
};
