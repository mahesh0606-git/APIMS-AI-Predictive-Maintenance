"""Demonstration scalarized evolutionary scheduler, not full NSGA-II."""
from dataclasses import dataclass
import random

@dataclass(frozen=True)
class Task:
    name: str
    urgency: float
    duration_hours: float
    part_cost: float

@dataclass(frozen=True)
class Assignment:
    day: int
    shift: int
    technician: int

def score(tasks, plan):
    """Illustrative risk/cost objective; NOT an SLA-validated cost model."""
    total=0.0
    for task, a in zip(tasks,plan):
        total+=task.urgency*(a.day+1)*task.duration_hours + .01*task.part_cost + (5 if a.shift==1 else 0)
    total+=1000*(len(plan)-len({(a.day,a.shift,a.technician) for a in plan}))
    return total

def optimize(tasks, days=14, technicians=3, population=40, generations=50, seed=7):
    """Return three lowest-score candidate plans using elitism/crossover/mutation."""
    if days<1 or technicians<1 or population<4 or generations<1:
        raise ValueError('Invalid scheduler parameters')
    if not tasks: return []
    rng=random.Random(seed)
    def gene(): return Assignment(rng.randrange(days),rng.randrange(2),rng.randrange(technicians))
    pop=[tuple(gene() for _ in tasks) for _ in range(population)]
    for _ in range(generations):
        pop.sort(key=lambda p:score(tasks,p))
        parents=pop[:max(2,population//2)]
        children=[]
        while len(parents)+len(children)<population:
            a,b=rng.sample(parents,2)
            cut=len(tasks)//2
            child=list(a[:cut]+b[cut:])
            if rng.random()<.25:
                child[rng.randrange(len(child))]=gene()
            children.append(tuple(child))
        pop=parents+children
    pop.sort(key=lambda p:score(tasks,p))
    return [(score(tasks,p),p) for p in pop[:3]]
