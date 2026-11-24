"""Explicit, fixed-width semantics and manually seeded first-order mutants.

These are standalone Python models, not implementations extracted from a product.
Inputs and arithmetic outputs are unsigned bit patterns. Signed comparisons use
two's-complement interpretation. Shift counts are reduced modulo the word width.
"""
from dataclasses import dataclass
from typing import Callable
import numpy as np

OPS = ("add", "sub", "mul", "and", "xor", "shl", "lsr", "asr", "ltu", "lts")


def signed(a: int, width: int) -> int:
    return a - (1 << width) if a & (1 << (width - 1)) else a


def reference(op: str, a: int, b: int, width: int) -> int:
    mask = (1 << width) - 1
    k = b % width
    if op == "add": return (a + b) & mask
    if op == "sub": return (a - b) & mask
    if op == "mul": return (a * b) & mask
    if op == "and": return a & b
    if op == "xor": return a ^ b
    if op == "shl": return (a << k) & mask
    if op == "lsr": return a >> k
    if op == "asr": return (signed(a, width) >> k) & mask
    if op == "ltu": return int(a < b)
    if op == "lts": return int(signed(a, width) < signed(b, width))
    raise ValueError(op)


@dataclass(frozen=True)
class Mutant:
    name: str
    op: str
    description: str
    fn: Callable[[int, int], int]


def make_mutants(width: int) -> list[Mutant]:
    if width not in (8, 16, 32):
        raise ValueError("Supported widths: 8, 16, 32")
    m = (1 << width) - 1
    sign = 1 << (width - 1)
    h = width // 2
    low = (1 << h) - 1
    s = lambda a: signed(a, width)
    r = lambda op, a, b: reference(op, a, b, width)
    result = []
    def add(op, suffix, desc, fn):
        result.append(Mutant(f"{op}.{suffix}", op, desc, fn))

    add("add", "no_wrap", "Omit output truncation", lambda a,b: a+b)
    add("add", "half_output", "Truncate output to half width", lambda a,b: (a+b)&low)
    add("add", "no_limb_carry", "Omit carry between half-width limbs", lambda a,b: (((a>>h)+(b>>h))<<h | ((a+b)&low))&m)
    add("add", "xor", "Replace addition by XOR", lambda a,b: a^b)
    add("add", "rhs_low", "Drop high half of second operand", lambda a,b: (a+(b&low))&m)
    add("add", "zero_lhs", "Wrong result in zero-lhs shortcut", lambda a,b: 0 if a==0 else r("add",a,b))
    add("add", "zero_rhs", "Wrong result in zero-rhs shortcut", lambda a,b: 0 if b==0 else r("add",a,b))
    add("add", "saturate", "Use signed saturation instead of wrapping", lambda a,b: min(sign-1,max(-sign,s(a)+s(b)))&m)

    add("sub", "no_wrap", "Omit output truncation", lambda a,b: a-b)
    add("sub", "swap", "Swap operands", lambda a,b: (b-a)&m)
    add("sub", "add", "Replace subtraction by addition", lambda a,b: (a+b)&m)
    add("sub", "no_borrow", "Omit borrow between half-width limbs", lambda a,b: ((((a>>h)-(b>>h))<<h)|((a-b)&low))&m)
    add("sub", "zero_rhs", "Wrong result in zero-rhs shortcut", lambda a,b: 0 if b==0 else r("sub",a,b))
    add("sub", "equal", "Wrong result for equal operands", lambda a,b: 1 if a==b else r("sub",a,b))
    add("sub", "saturate", "Use unsigned saturation", lambda a,b: max(0,a-b))

    add("mul", "no_wrap", "Omit output truncation", lambda a,b: a*b)
    add("mul", "add", "Replace multiplication by addition", lambda a,b: (a+b)&m)
    add("mul", "lhs_low", "Drop high half of first operand", lambda a,b: ((a&low)*b)&m)
    add("mul", "high_half", "Return high instead of low product", lambda a,b: (a*b)>>width)
    add("mul", "zero", "Wrong zero-product shortcut", lambda a,b: 1 if a==0 or b==0 else r("mul",a,b))
    add("mul", "one", "Wrong multiplicative-identity shortcut", lambda a,b: 0 if a==1 or b==1 else r("mul",a,b))
    add("mul", "saturate", "Use unsigned saturation", lambda a,b: min(m,a*b))

    add("and", "or", "Replace AND by OR", lambda a,b: a|b)
    add("and", "xor", "Replace AND by XOR", lambda a,b: a^b)
    add("and", "invert_lhs", "Invert first operand", lambda a,b: (~a)&b&m)
    add("and", "half_output", "Drop upper output bits", lambda a,b: (a&b)&low)
    add("and", "lhs", "Ignore second operand", lambda a,b: a)
    add("and", "zero_rhs", "Wrong zero-rhs shortcut", lambda a,b: a if b==0 else r("and",a,b))

    add("xor", "or", "Replace XOR by OR", lambda a,b: a|b)
    add("xor", "and", "Replace XOR by AND", lambda a,b: a&b)
    add("xor", "invert", "Invert result", lambda a,b: (~(a^b))&m)
    add("xor", "half_output", "Drop upper output bits", lambda a,b: (a^b)&low)
    add("xor", "lhs", "Ignore second operand", lambda a,b: a)
    add("xor", "equal", "Wrong equal-input shortcut", lambda a,b: a if a==b else r("xor",a,b))

    for op in ("shl", "lsr", "asr"):
        add(op, "half_count", "Use an undersized shift-count mask", lambda a,b,op=op: r(op,a,b&(h-1)))
        add(op, "off_by_one", "Increment masked shift count", lambda a,b,op=op: r(op,a,(b+1)%width))
        add(op, "zero_count", "Wrong zero-shift shortcut", lambda a,b,op=op: 0 if b%width==0 else r(op,a,b))
        add(op, "large_count", "Return zero rather than mask large count", lambda a,b,op=op: 0 if b>=width else r(op,a,b))
    add("shl", "no_wrap", "Omit left-shift truncation", lambda a,b: a<<(b%width))
    add("shl", "direction", "Shift right instead of left", lambda a,b: a>>(b%width))
    add("lsr", "signed", "Arithmetic instead of logical right shift", lambda a,b: r("asr",a,b))
    add("lsr", "direction", "Shift left instead of right", lambda a,b: r("shl",a,b))
    add("asr", "unsigned", "Logical instead of arithmetic right shift", lambda a,b: a>>(b%width))
    add("asr", "direction", "Shift left instead of right", lambda a,b: r("shl",a,b))
    add("asr", "wrong_sign", "Subtract wrong sign-conversion offset", lambda a,b: ((a-sign if a>=sign else a)>>(b%width))&m)

    add("ltu", "signed", "Compare signed operands", lambda a,b: int(s(a)<s(b)))
    add("ltu", "inclusive", "Use less-than-or-equal", lambda a,b: int(a<=b))
    add("ltu", "reverse", "Reverse comparison", lambda a,b: int(a>b))
    add("ltu", "half", "Compare only low halves", lambda a,b: int((a&low)<(b&low)))
    add("ltu", "false", "Always return false", lambda a,b: 0)
    add("ltu", "zero_lhs", "Wrong zero-lhs shortcut", lambda a,b: 0 if a==0 else r("ltu",a,b))
    add("lts", "unsigned", "Compare unsigned operands", lambda a,b: int(a<b))
    add("lts", "inclusive", "Use less-than-or-equal", lambda a,b: int(s(a)<=s(b)))
    add("lts", "reverse", "Reverse comparison", lambda a,b: int(s(a)>s(b)))
    add("lts", "half", "Compare signed low halves", lambda a,b: int(signed(a&low,h)<signed(b&low,h)))
    add("lts", "false", "Always return false", lambda a,b: 0)
    add("lts", "zero_rhs", "Wrong zero-rhs shortcut", lambda a,b: 0 if b==0 else r("lts",a,b))
    return result


def choose_mutants(all_mutants, count, seed):
    """Round-robin operations so small subsets do not omit entire operations."""
    rng = np.random.default_rng(seed)
    groups = []
    for op in OPS:
        ids = [i for i,m in enumerate(all_mutants) if m.op==op]
        rng.shuffle(ids)
        groups.append(ids)
    selected = []
    for k in range(max(map(len, groups))):
        for group in groups:
            if k < len(group): selected.append(group[k])
    return sorted(selected[:min(count,len(selected))])


def generate_tests(n, width, boundary_ratio, seed):
    rng = np.random.default_rng(seed)
    m = (1<<width)-1
    sign = 1<<(width-1)
    half = 1<<(width//2)
    # Fixed, operation-independent diagnostic prefix, not chosen using kill data.
    pairs = [(0,1),(1,0),(1,1),(m,1),(sign-1,1),(sign,1),(m,m),
             (sign,sign),(half-1,1),(m,0),(0,m),(m,width),
             (sign,width+1),(m,width-1),(2,2),(1,width),(m,width+1),(sign,0)]
    tests = [(op,a,b) for a,b in pairs for op in OPS]
    if n < len(tests): raise ValueError(f"n must be >= {len(tests)}")
    edges = np.array(sorted({0,1,2,3,half-1,half,half+1,sign-1,sign,sign+1,m-1,m,width-1,width,width+1}),dtype=np.int64)
    while len(tests)<n:
        op = OPS[int(rng.integers(len(OPS)))]
        if rng.random()<boundary_ratio:
            a,b = map(int,rng.choice(edges,2))
            if op in ("ltu","lts","sub") and rng.random()<0.35:
                b = (a+int(rng.choice([-1,0,1])))&m
        else:
            a,b = map(int,rng.integers(0,m+1,size=2,dtype=np.int64))
        tests.append((op,a,b))
    return tests


def kill_matrix(tests, mutants, width):
    matrix = np.zeros((len(mutants),len(tests)),dtype=np.uint8)
    groups = {op:[(i,m) for i,m in enumerate(mutants) if m.op==op] for op in OPS}
    for j,(op,a,b) in enumerate(tests):
        expected = reference(op,a,b,width)
        for i,mutant in groups[op]:
            matrix[i,j] = mutant.fn(a,b)!=expected
    return matrix


def synthetic_matrix(m,n,density,seed):
    """Independent Bernoulli stress matrix; not an arithmetic execution result.

    Any uncovered row gets one random witness, solely to ensure feasibility.
    """
    rng=np.random.default_rng(seed)
    a=(rng.random((m,n))<density).astype(np.uint8)
    missing=np.where(a.sum(axis=1)==0)[0]
    for i in missing: a[i,int(rng.integers(n))]=1
    return a
