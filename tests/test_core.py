"""Checks semantic oracles, optimality, preprocessing, and replayable coverage."""
import itertools
import unittest
import numpy as np
from atm.benchmark import OPS,reference,generate_tests,make_mutants,kill_matrix
from atm.solvers import compress,greedy,lp_round,integer_program,validate


def ripple_add(a,b,width):
    carry=0;result=0
    for bit in range(width):
        x=(a>>bit)&1;y=(b>>bit)&1
        result|=(x^y^carry)<<bit
        carry=(x&y)|(x&carry)|(y&carry)
    return result


def brute_cover(a):
    for k in range(a.shape[1]+1):
        for chosen in itertools.combinations(range(a.shape[1]),k):
            if validate(a,list(chosen)): return k
    return None


class CoreChecks(unittest.TestCase):
    def test_exhaustive_8bit_arithmetic_against_bit_algorithms(self):
        for a,b in itertools.product(range(256),repeat=2):
            self.assertEqual(reference("add",a,b,8),ripple_add(a,b,8))
            self.assertEqual(reference("sub",a,b,8),ripple_add(a,ripple_add(b^255,1,8),8))
            product=0
            for bit in range(8):
                if (b>>bit)&1: product=ripple_add(product,(a<<bit)&255,8)
            self.assertEqual(reference("mul",a,b,8),product)

    def test_signed_shift_and_comparison_boundaries(self):
        for w in (8,16,32):
            m=(1<<w)-1;s=1<<(w-1)
            self.assertEqual(reference("asr",m,1,w),m)
            self.assertEqual(reference("lsr",m,1,w),s-1)
            self.assertEqual(reference("lts",s,0,w),1)
            self.assertEqual(reference("ltu",s,0,w),0)
            self.assertEqual(reference("shl",1,w,w),1)
            self.assertEqual(reference("shl",s,1,w),0)

    def test_every_seeded_mutant_has_an_observed_witness(self):
        for width in (8,16,32):
            mutants=make_mutants(width)
            self.assertEqual(len({m.name for m in mutants}),len(mutants))
            tests=generate_tests(1200,width,0.5,11)
            a=kill_matrix(tests,mutants,width)
            missing=[m.name for m,hit in zip(mutants,a.any(axis=1)) if not hit]
            self.assertEqual(missing,[])

    def test_solver_optima_against_exhaustive_enumeration(self):
        rng=np.random.default_rng(2026)
        for _ in range(30):
            a=(rng.random((6,9))<0.4).astype(np.uint8)
            a[:,0]=1 if not a.any() else a[:,0]
            for i in np.where(~a.any(axis=1))[0]:a[i,int(rng.integers(9))]=1
            optimum=brute_cover(a)
            ip=integer_program(a)
            self.assertEqual(ip.info["status"],"optimal")
            self.assertEqual(len(ip.selected),optimum)
            lp=lp_round(a,seed=9)
            self.assertLessEqual(lp.info["lp_bound"],optimum+1e-6)
            for result in (greedy(a),lp,ip):
                self.assertTrue(validate(a,result.selected))
                self.assertGreaterEqual(len(result.selected),optimum)

    def test_fractional_triangle_and_duplicate_columns(self):
        a=np.array([[1,1,0,0,1],[1,0,1,0,1],[0,1,1,0,0]],dtype=np.uint8)
        b,columns,active=compress(a)
        self.assertEqual(b.shape,(3,3))
        self.assertEqual(len(integer_program(b).selected),2)
        lp=lp_round(b,seed=3)
        self.assertAlmostEqual(lp.info["lp_bound"],1.5)
        self.assertTrue(validate(a[active],columns[lp.selected].tolist()))

    def test_generation_is_reproducible(self):
        self.assertEqual(generate_tests(300,32,.2,11),generate_tests(300,32,.2,11))
        self.assertNotEqual(generate_tests(300,32,.2,11),generate_tests(300,32,.2,23))


if __name__=="__main__":unittest.main()
