import unittest
import torch
from mergers import ties, joint_basis, knots_ties, match_norm


class MergerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(123)

    def test_sign_election_and_disjoint_mean(self):
        a = torch.tensor([4., -4., 0., 2., -2.])
        b = torch.tensor([-1., -2., 3., 6., 1.])
        torch.testing.assert_close(ties([a,b],1),torch.tensor([4.,-3.,3.,4.,-2.]))
        torch.testing.assert_close(ties([a,b],1),ties([b,a],1))

    def test_threshold_ties_and_zeros(self):
        a = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
        # Official kthvalue threshold retains 4 entries for d=10,K=.3.
        torch.testing.assert_close(ties([a],.3),torch.tensor([0.,0.,0.,0.,0.,0.,7.,8.,9.,10.]))
        torch.testing.assert_close(ties([a,-a],1),-a)
        torch.testing.assert_close(ties([torch.zeros(10)]),torch.zeros(10))

    def test_thin_svd_matches_dense_and_linear(self):
        ups = [torch.randn(17,3), torch.randn(17,5)]
        downs = [torch.randn(3,13), torch.randn(5,13)]
        left, blocks, s = joint_basis(ups,downs)
        dense = [b@a for b,a in zip(ups,downs)]
        for block, d in zip(blocks,dense):
            torch.testing.assert_close(left@block,d,atol=3e-6,rtol=3e-6)
        torch.testing.assert_close(left@sum(blocks),sum(dense),atol=5e-6,rtol=3e-6)
        exact = torch.cat([b.double()@a.double() for b,a in zip(ups,downs)],1)
        u, sd, vh = torch.linalg.svd(exact,full_matrices=False)
        u, vh, sd = u[:,:len(s)].float(), vh[:len(s)].float(), sd[:len(s)].float()
        ref = u@ties(list((sd[:,None]*vh).split(13,1)),.3)
        actual, rank = knots_ties(ups,downs,.3)
        self.assertEqual(rank,8)
        torch.testing.assert_close(s,sd)
        torch.testing.assert_close(actual,ref,atol=4e-6,rtol=3e-6)

    def test_basis_sign_invariance_and_norm(self):
        ups=[torch.randn(15,3),torch.randn(15,3)]
        downs=[torch.randn(3,12),torch.randn(3,12)]
        left, blocks, _=joint_basis(ups,downs)
        signs=torch.tensor([1.,-1.,1.,-1.,1.,-1.])
        torch.testing.assert_close(left@ties(blocks), (left*signs)@ties([x*signs[:,None] for x in blocks]))
        ref=sum(b@a for b,a in zip(ups,downs))
        result, metrics=match_norm(left@ties(blocks),ref)
        torch.testing.assert_close(result.norm(),ref.norm())
        self.assertGreater(metrics['scale'],0)
        with self.assertRaises(ValueError):match_norm(torch.zeros_like(ref),ref)


if __name__ == '__main__':
    unittest.main()
