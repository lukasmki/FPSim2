from rdkit.Chem import rdChemReactions
from rdkit.DataStructs import ExplicitBitVect
from .FPSim2 import FPSim2Engine
from .io.chem import build_rxn_fp, process_fp, load_reaction
from typing import Union
import numpy as np


class ReactionEngine(FPSim2Engine):
    """FPSim2Engine subclass for reaction fingerprint similarity search.

    Accepts reaction SMARTS strings or ChemicalReaction objects as queries.
    All search methods (similarity, tversky, substructure, top_k, etc.)
    are inherited unchanged from FPSim2Engine.

    The HDF5 database must be created with create_reaction_db_file().
    """

    def load_query(
        self,
        query: Union[str, ExplicitBitVect, "rdChemReactions.ChemicalReaction"],
        full_sanitization: bool = True,
        mol_format: str = None,
    ) -> np.ndarray:
        """Loads the query fingerprint from a reaction SMARTS string,
        a ChemicalReaction object, or an ExplicitBitVect.

        Parameters
        ----------
        query : str, ExplicitBitVect, or ChemicalReaction
            Reaction SMARTS string, RDKit ChemicalReaction object,
            or pre-computed ExplicitBitVect fingerprint.
        full_sanitization : bool
            Ignored (kept for signature compatibility with BaseEngine).
        """
        if isinstance(query, ExplicitBitVect):
            fp = process_fp(query, 0)
        elif isinstance(query, rdChemReactions.ChemicalReaction):
            fp = build_rxn_fp(query, self.fp_type, self.fp_params, 0)
        else:
            rxn = load_reaction(query)
            fp = build_rxn_fp(rxn, self.fp_type, self.fp_params, 0)
        return np.array(fp, dtype=np.uint64)
