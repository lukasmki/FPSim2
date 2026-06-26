from FPSim2.io.backends.pytables import create_reaction_db_file
from FPSim2.io.chem import (
    sma_rxn_supplier,
    it_rxn_supplier,
    get_rxn_supplier,
)
from FPSim2 import ReactionEngine
from rdkit.Chem import rdChemReactions
from rdkit.DataStructs import ExplicitBitVect
import numpy as np
import pytest
import os

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SMA_FILE = os.path.join(TESTS_DIR, "data/10rxns.sma")
RXN_H5 = os.path.join(TESTS_DIR, "data/10rxns.h5")

# First reaction from 10rxns.sma (line 1)
QUERY_SMARTS = "[#6:1][N:2]=[C:3]=[S:4].[F,Cl,Br,I][C:5][C;!$(C(=O)[N,O,S,F,Cl,Br,I]):6]=O.[NH2;$([N][#6]);!$([N]C=[O,S,N]):7]>>[#6:1][N:2]=[c:3]1[n:7][c:6][c:5][s:4]1"


# ---- Supplier tests ----


def test_sma_rxn_supplier_count():
    rxns = list(sma_rxn_supplier(SMA_FILE))
    # line 10 is '>>' (empty reaction) — may be included or skipped depending on RDKit
    assert 9 <= len(rxns) <= 10


def test_sma_rxn_supplier_ids_default_to_lineno():
    rxns = list(sma_rxn_supplier(SMA_FILE))
    ids = [rxn_id for rxn_id, _ in rxns]
    assert ids[0] == 1
    assert all(isinstance(i, int) for i in ids)


def test_sma_rxn_supplier_yields_chemicalreaction():
    for rxn_id, rxn in sma_rxn_supplier(SMA_FILE):
        assert isinstance(rxn, rdChemReactions.ChemicalReaction)


def test_get_rxn_supplier_sma():
    assert get_rxn_supplier(SMA_FILE) is sma_rxn_supplier


def test_get_rxn_supplier_iterable():
    assert get_rxn_supplier([(QUERY_SMARTS, 1)]) is it_rxn_supplier


def test_it_rxn_supplier_smarts():
    data = [
        (QUERY_SMARTS, 1),
        ("[NH2:1][NH:2][#6:3].[#6:4][CH:5]=O>>[#6:4][CH:5]=[N:1][N:2][#6:3]", 2),
    ]
    rxns = list(it_rxn_supplier(data))
    assert len(rxns) == 2
    assert rxns[0][0] == 1
    assert isinstance(rxns[0][1], rdChemReactions.ChemicalReaction)


def test_it_rxn_supplier_rxn_object():
    rxn_obj = rdChemReactions.ReactionFromSmarts(QUERY_SMARTS)
    rxns = list(it_rxn_supplier([(rxn_obj, 99)]))
    assert len(rxns) == 1
    assert rxns[0][0] == 99
    assert isinstance(rxns[0][1], rdChemReactions.ChemicalReaction)


def test_it_rxn_supplier_invalid_id_raises():
    with pytest.raises(Exception, match="integer ids"):
        list(it_rxn_supplier([(QUERY_SMARTS, "not_an_int")]))


# ---- DB creation tests ----


@pytest.mark.incremental
class TestCreateReactionDB:
    def test_create_reaction_db_file_sma(self):
        create_reaction_db_file(SMA_FILE, RXN_H5)
        engine = ReactionEngine(RXN_H5)
        assert 9 <= engine.fps.shape[0] <= 10
        assert engine.fp_type == "RDKitPattern"
        assert engine.fp_params["fpSize"] == 2048

    def test_create_reaction_db_file_iterable(self, tmp_path):
        out_h5 = str(tmp_path / "rxns_list.h5")
        data = [
            (QUERY_SMARTS, 1),
            ("[NH2:1][NH:2][#6:3].[#6:4][CH:5]=O>>[#6:4][CH:5]=[N:1][N:2][#6:3]", 2),
        ]
        create_reaction_db_file(data, out_h5)
        engine = ReactionEngine(out_h5)
        assert engine.fps.shape[0] == 2

    def test_create_reaction_db_file_custom_fp_params(self, tmp_path):
        out_h5 = str(tmp_path / "rxns_custom.h5")
        create_reaction_db_file(SMA_FILE, out_h5, fp_params={"fpSize": 4096})
        engine = ReactionEngine(out_h5)
        assert engine.fp_params["fpSize"] == 4096

    def test_create_reaction_db_file_invalid_fp_type_raises(self, tmp_path):
        out_h5 = str(tmp_path / "bad.h5")
        with pytest.raises(ValueError, match="Unsupported fp_type"):
            create_reaction_db_file(SMA_FILE, out_h5, fp_type="NotAType")

    def test_reaction_db_sorted_by_popcnt(self):
        engine = ReactionEngine(RXN_H5)
        popcnts = engine.fps[:, -1]
        assert (popcnts[:-1] <= popcnts[1:]).all()


# ---- ReactionEngine.load_query() tests ----


class TestReactionEngineLoadQuery:
    def test_load_query_smarts_string(self):
        engine = ReactionEngine(RXN_H5)
        q = engine.load_query(QUERY_SMARTS)
        assert isinstance(q, np.ndarray)
        assert q.dtype == np.uint64
        # 1 (id) + 32 (fp uint64 chunks for 2048 bits) + 1 (popcnt) = 34
        assert q.shape == (34,)

    def test_load_query_chemicalreaction_matches_string(self):
        engine = ReactionEngine(RXN_H5)
        rxn_obj = rdChemReactions.ReactionFromSmarts(QUERY_SMARTS)
        q_str = engine.load_query(QUERY_SMARTS)
        q_obj = engine.load_query(rxn_obj)
        np.testing.assert_array_equal(q_str, q_obj)

    def test_load_query_explicithbitvect(self):
        engine = ReactionEngine(RXN_H5)
        rxn = rdChemReactions.ReactionFromSmarts(QUERY_SMARTS)
        params = rdChemReactions.ReactionFingerprintParams()
        params.fpType = rdChemReactions.FingerprintType.PatternFP
        params.fpSize = 2048
        efp = rdChemReactions.CreateStructuralFingerprintForReaction(rxn, params)
        assert isinstance(efp, ExplicitBitVect)
        q = engine.load_query(efp)
        assert isinstance(q, np.ndarray)
        assert q.dtype == np.uint64


# ---- Search method tests ----


class TestReactionEngineSearch:
    def test_similarity_returns_results(self):
        engine = ReactionEngine(RXN_H5)
        results = engine.similarity(QUERY_SMARTS, 0.1)
        assert results.shape[0] > 0
        assert "mol_id" in results.dtype.names
        assert "coeff" in results.dtype.names

    def test_similarity_self_query_at_threshold_1(self):
        engine = ReactionEngine(RXN_H5)
        results = engine.similarity(QUERY_SMARTS, 1.0)
        assert any(r["mol_id"] == 1 for r in results)
        assert any(abs(r["coeff"] - 1.0) < 1e-5 for r in results if r["mol_id"] == 1)

    def test_similarity_chemicalreaction_matches_smarts(self):
        engine = ReactionEngine(RXN_H5)
        rxn_obj = rdChemReactions.ReactionFromSmarts(QUERY_SMARTS)
        results_str = engine.similarity(QUERY_SMARTS, 0.1)
        results_obj = engine.similarity(rxn_obj, 0.1)
        np.testing.assert_array_equal(results_str, results_obj)

    def test_substructure_returns_array(self):
        engine = ReactionEngine(RXN_H5)
        results = engine.substructure(QUERY_SMARTS)
        assert isinstance(results, np.ndarray)

    def test_top_k_returns_at_most_k(self):
        engine = ReactionEngine(RXN_H5)
        results = engine.top_k(QUERY_SMARTS, k=3, threshold=0.0)
        assert results.shape[0] <= 3

    def test_on_disk_similarity(self):
        engine = ReactionEngine(RXN_H5, in_memory_fps=False)
        results = engine.on_disk_similarity(QUERY_SMARTS, 0.1)
        assert results.shape[0] > 0

    def test_tversky(self):
        engine = ReactionEngine(RXN_H5)
        results = engine.tversky(QUERY_SMARTS, 0.1, 0.5, 0.5)
        assert isinstance(results, np.ndarray)

    def test_n_workers_parallel_matches_single(self):
        engine = ReactionEngine(RXN_H5)
        r1 = engine.similarity(QUERY_SMARTS, 0.1, n_workers=1)
        r2 = engine.similarity(QUERY_SMARTS, 0.1, n_workers=2)
        np.testing.assert_array_equal(r1, r2)


def teardown_module(module):
    if os.path.exists(RXN_H5):
        os.remove(RXN_H5)
