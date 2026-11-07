# CREDITS

Everything this project uses that someone else made: papers, methods, code, data and models.
Generated from the encyclopedia data (`site/enc/data/part_*.json`, by `site/enc/make_credits.py`) on 23 Oct 2026; every
module in `src/nds/` also credits its own sources in its docstring.

Rule: we re-implement from papers. No code from GPL, Yale non-commercial or unlicensed
repositories is copied into this repo; permissive (MIT/BSD/Apache) code is adapted only with
credit at the place of use. See `docs/code-notes.md` section 1.

## I. Spectral graph theory and the numerical tools

- **Spectral clustering tutorial**: von Luxburg, U. (2007). A tutorial on spectral clustering. Statistics and Computing 17(4): 395-416.
  [arXiv](https://arxiv.org/abs/0711.0189) · [PDF (author page)](https://www.tml.cs.uni-tuebingen.de/team/luxburg/publications/Luxburg07_tutorial.pdf) · [DOI](https://doi.org/10.1007/s11222-007-9033-z)
  Used in: `nds.spectra.eigengap_module_count`, `nds.spectra.diffusion_spectrum`, `nds.modules.spectral_partition`, `docs/method-notes.md §3`, `DECISIONS.md D15`
- **Chung's normalized Laplacian and Cheeger**: Chung, F. R. K. (1997). Spectral Graph Theory. CBMS Regional Conference Series in Mathematics, vol. 92. American Mathematical Society.
  [Ch. 1 (revised, free)](https://fanchung.ucsd.edu/research/cb/ch1.pdf) · [Ch. 2 (revised, free)](https://fanchung.ucsd.edu/research/cb/ch2.pdf)
  Used in: `nds.modules.conductances`, `nds.spectra.diffusion_spectrum`
  (verification: partly; see the research files)
- **Higher-order Cheeger inequality**: Lee, J. R., Oveis Gharan, S., Trevisan, L. (2014). Multiway spectral partitioning and higher-order Cheeger inequalities. Journal of the ACM 61(6): 37:1-37:30. Conference version: Proc. ACM STOC 2012, pp. 1117-1130.
  [arXiv](https://arxiv.org/abs/1111.1055) · [DOI](https://doi.org/10.1145/2665063)
  Used in: `nds.modules.conductances`, `nds.modules.upsilon`, `nds.spectra.eigengap_module_count`
- **Upsilon: when spectral clustering works**: Peng, R., Sun, H., Zanetti, L. (2017). Partitioning well-clustered graphs: spectral clustering works! SIAM Journal on Computing 46(2): 710-743. Preliminary version: COLT 2015.
  [arXiv](https://arxiv.org/abs/1411.2021) · [PDF (journal version, author page)](https://homepages.inf.ed.ac.uk/hsun4/SICOMP.pdf) · [DOI](https://doi.org/10.1137/15M1047209) · [STAG library (He Sun's group)](https://staglibrary.io/) · [Follow-up: Tyler & Zanetti 2025 (preprint)](https://arxiv.org/abs/2511.23261)
  Used in: `nds.modules.upsilon`, `nds.modules.spectral_partition`, `nds.modules.conductances`, `nds.analysis.spectrum_numbers`, `docs/method-notes.md §3`, `DECISIONS.md D15`
- **Spielman's textbook (MAT 6495)**: Spielman, D. A. (2025). Spectral and Algebraic Graph Theory. Book draft, title page dated April 2, 2025, free PDF on the author's site.
  [Book PDF](https://www.cs.yale.edu/homes/spielman/sagt/sagt.pdf) · [Book page](https://www.cs.yale.edu/homes/spielman/sagt) · [Code (Julia notebook)](https://github.com/danspielman/sagt_code) · [Lecture notes 2015 (Cheeger)](https://www.cs.yale.edu/homes/spielman/561/lect06-15.pdf) · [Lecture notes 2018 (Cheeger)](https://www.cs.yale.edu/homes/spielman/561/lect11-18.pdf)
  Used in: `nds.spectra.symmetric_diffusion_operator`, `nds.spectra.diffusion_spectral_entropy`, `nds.modules.conductances`
  (verification: partly; see the research files)
- **Wolf's course MAT 6495**: Wolf, G. (2026). MAT 6495: Spectral Graph Theory. Graduate course (4 credits), Universite de Montreal, also offered through the ISM.
  [Course page](https://mat6495.wolf.courses)
- **Self-tuning kernel (and alpha-decay)**: Zelnik-Manor, L., Perona, P. (2004). Self-tuning spectral clustering. Advances in Neural Information Processing Systems 17 (NIPS 2004).
  [PDF (NeurIPS)](https://papers.nips.cc/paper_files/paper/2004/file/40173ea48d9567f1f393b20c855bb40b-Paper.pdf) · [Paper page](https://lihi.net.technion.ac.il/?p=112) · [Matlab code (zip, not checked)](http://lihi.eew.technion.ac.il/files/Demos/SelfTuning/ZPclustering.zip) · [Alpha-decay kernel: PHATE (DOI)](https://doi.org/10.1038/s41587-019-0336-3) · [Alpha-decay kernel code: graphtools](https://github.com/KrishnaswamyLab/graphtools)
  Used in: `nds.graphs.neighbour_bandwidths`, `nds.graphs.adaptive_gaussian_kernel`, `nds.graphs.correlation_distance_squared`, `nds.graphs.neuron_graph_from_covariance`, `tests/test_graphs_and_spectra.py`, `DECISIONS.md D16`
  (verification: partly; see the research files)
- **Weyl and Davis-Kahan bounds**: Weyl, H. (1912). Das asymptotische Verteilungsgesetz der Eigenwerte linearer partieller Differentialgleichungen (mit einer Anwendung auf die Theorie der Hohlraumstrahlung). Mathematische Annalen 71(4): 441-479. Davis, C., Kahan, W. M. (1970). The rotation of eigenvectors by a perturbation. III. SIAM Journal on Numerical Analysis 7(1): 1-46. doi:10.1137/0707001.
  [DOI (Weyl 1912)](https://doi.org/10.1007/BF01456804) · [DOI (Davis-Kahan 1970)](https://doi.org/10.1137/0707001)
  Used in: `nds.spectra.davis_kahan_ratio`
  (verification: partly; see the research files)
- **Yu-Wang-Samworth sin-theta bound**: Yu, Y., Wang, T., Samworth, R. J. (2015). A useful variant of the Davis-Kahan theorem for statisticians. Biometrika 102(2): 315-323.
  [arXiv](https://arxiv.org/abs/1405.0680) · [PDF (author page)](https://personal.lse.ac.uk/wangt60/publication/DKvariant.pdf) · [DOI](https://doi.org/10.1093/biomet/asv008)
  Used in: `nds.spectra.subspace_localization`, `nds.flow.match_modes`, `nds.spectra.davis_kahan_ratio`
- **IPR and localization**: Bell, R. J., Dean, P. (1970). Atomic vibrations in vitreous silica. Discussions of the Faraday Society 50: 55. Evers, F., Mirlin, A. D. (2008). Anderson transitions. Reviews of Modern Physics 80: 1355. Goltsev, A. V., Dorogovtsev, S. N., Oliveira, J. G., Mendes, J. F. F. (2012). Localization and spreading of diseases in complex networks. Physical Review Letters 109: 128702.
  [DOI (Bell & Dean 1970)](https://doi.org/10.1039/DF9705000055) · [arXiv (Evers & Mirlin)](https://arxiv.org/abs/0707.4378) · [DOI (Evers & Mirlin)](https://doi.org/10.1103/RevModPhys.80.1355) · [arXiv (Goltsev et al.)](https://arxiv.org/abs/1202.4411) · [DOI (Goltsev et al.)](https://doi.org/10.1103/PhysRevLett.109.128702) · [arXiv (Martin, Zhang & Newman 2014)](https://arxiv.org/abs/1401.5093)
  Used in: `nds.spectra.inverse_participation_ratio`, `nds.spectra.localization_ratio`, `nds.spectra.subspace_localization`, `nds.spectra.spectrum_summary`, `nds.controls.rotate_live_covariance`, `docs/method-notes.md §2`, `DECISIONS.md D14`
  (verification: partly; see the research files)
- **Hungarian mode matching**: Kuhn, H. W. (1955). The Hungarian method for the assignment problem. Naval Research Logistics Quarterly 2(1-2): 83-97.
  [DOI](https://doi.org/10.1002/nav.3800020109) · [DOI (Crouse 2016, the algorithm scipy uses)](https://doi.org/10.1109/TAES.2016.140952)
  Used in: `nds.flow.match_modes`, `nds.flow.restrict_to_common_neurons`, `nds.timeline.spectral_flow`, `tests/test_flow.py`
- **Avoided crossings, Landau-Zener**: von Neumann, J., Wigner, E. (1929). Ueber das Verhalten von Eigenwerten bei adiabatischen Prozessen. Physikalische Zeitschrift 30: 467-470. Zener, C. (1932). Non-adiabatic crossing of energy levels. Proceedings of the Royal Society of London A 137: 696-702. Shevchenko, S. N., Ashhab, S., Nori, F. (2010). Landau-Zener-Stueckelberg interferometry. Physics Reports 492(1).
  [DOI (Zener 1932)](https://doi.org/10.1098/rspa.1932.0165) · [arXiv (Shevchenko et al. 2010)](https://arxiv.org/abs/0911.1917) · [Trefethen 2026, Eigenvalue repulsion (PDF)](https://people.maths.ox.ac.uk/trefethen/repulsion.pdf) · [arXiv (Frame et al. 2018, eigenvector continuation)](https://arxiv.org/abs/1711.07090) · [arXiv (Uhlig 2020, crossing eigencurves)](https://arxiv.org/abs/2002.01274) · [arXiv (Xu 2026, level crossings in training, preprint)](https://arxiv.org/abs/2603.28964)
  Used in: `nds.flow.two_level_eigenvalues`, `nds.flow.fit_avoided_crossing`, `nds.timeline.avoided_crossing_candidates`, `nds.timeline.training_time`, `nds.synthetic.three_module_path`, `tests/test_flow.py`, `tests/test_timeline.py`
  (verification: partly; see the research files)
- **Eigensolvers (eigh, eigsh, lobpcg)**: Lehoucq, R. B., Sorensen, D. C., Yang, C. (1998). ARPACK Users' Guide: Solution of Large-Scale Eigenvalue Problems with Implicitly Restarted Arnoldi Methods. SIAM. Knyazev, A. V. (2001). Toward the optimal preconditioned eigensolver: Locally Optimal Block Preconditioned Conjugate Gradient method. SIAM J. Sci. Comput. 23(2): 517-541.
  [Docs: scipy.linalg.eigh](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html) · [Docs: scipy.sparse.linalg.eigsh](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html) · [Docs: scipy.sparse.linalg.lobpcg](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lobpcg.html) · [DOI (Knyazev 2001)](https://doi.org/10.1137/S1064827500366124)
  Used in: `nds.spectra.diffusion_spectrum`, `nds.spectra.symmetric_diffusion_operator`, `tests/test_graphs_and_spectra.py`
- **PELT change-point detection**: Killick, R., Fearnhead, P., Eckley, I. A. (2012). Optimal detection of changepoints with a linear computational cost. Journal of the American Statistical Association 107(500): 1590-1598.
  [arXiv](https://arxiv.org/abs/1101.1438) · [Supplementary material](https://www.lancaster.ac.uk/~killick/Pub/KillickFearnheadEckley2012SuppMat.pdf) · [DOI](https://doi.org/10.1080/01621459.2012.737745) · [Code (R package changepoint)](https://github.com/rkillick/changepoint) · [CRAN](https://CRAN.R-project.org/package=changepoint) · [Package paper (J. Stat. Softw. 58(3))](http://www.jstatsoft.org/v58/i03/)
  Used in: `nds.timeline.change_points`
- **ruptures change-point library**: Truong, C., Oudre, L., Vayatis, N. (2020). Selective review of offline change point detection methods. Signal Processing 167: 107299.
  [arXiv](https://arxiv.org/abs/1801.00718) · [PDF (author)](http://www.laurentoudre.fr/publis/TOG-SP-19.pdf) · [DOI](https://doi.org/10.1016/j.sigpro.2019.107299) · [Code (ruptures)](https://github.com/deepcharles/ruptures) · [Docs (older)](https://ctruong.perso.math.cnrs.fr/ruptures-docs/build/html/) · [Notebook: basic usage](https://github.com/deepcharles/ruptures/blob/master/docs/examples/basic-usage.ipynb)
  Used in: `nds.timeline.change_points`
- **Pairwise (streaming) variance update**: Chan, T. F., Golub, G. H., LeVeque, R. J. (1983). Algorithms for computing the sample variance: analysis and recommendations. The American Statistician 37(3): 242-247.
  [PDF (course mirror)](https://www.math.pku.edu.cn/teachers/litj/notes/numer_anal/AmerStat_37_242_Chan_Variance.pdf) · [Reference (Praat manual)](https://www.fon.hum.uva.nl/praat/manual/Chan__Golub___LeVeque__1983_.html)
  Used in: `nds.moments.NeuronMoments`, `nds.moments.BlockedNeuronMoments`, `tests/test_moments.py`
  (verification: partly; see the research files)

## II. The diffusion lineage: Wolf, Krishnaswamy, Coifman and Mila

- **Diffusion maps and the alpha family**: Coifman, R. R., & Lafon, S. (2006). Diffusion maps. Applied and Computational Harmonic Analysis, 21(1), 5-30.
  [PDF (course mirror)](https://www.math.ucdavis.edu/~strohmer/courses/270/diffusion_maps.pdf) · [DOI](https://doi.org/10.1016/j.acha.2006.04.006)
  Used in: `nds.graphs.apply_anisotropy`, `nds.spectra.symmetric_diffusion_operator`, `nds.spectra.diffusion_spectrum`
- **The spectrum as slow physical modes**: Nadler, B., Lafon, S., Coifman, R. R., & Kevrekidis, I. G. (2006). Diffusion maps, spectral clustering and reaction coordinates of dynamical systems. Applied and Computational Harmonic Analysis, 21(1), 113-127.
  [arXiv PDF](https://arxiv.org/pdf/math/0503445) · [DOI](https://doi.org/10.1016/j.acha.2005.07.004) · [Short NeurIPS 2005 version](https://proceedings.neurips.cc/paper/2005/hash/2a0f97f81755e2878b264adf39cba68e-Abstract.html) · [Talk (MLSS 2005, Nadler)](https://new.videolectures.net/videos/mlss05us_nadler_dmscr)
  Used in: `nds.spectra.eigengap_module_count`, `docs/method-notes.md §3`
  (verification: partly; see the research files)
- **graphtools, the house kernel builder**: Krishnaswamy Lab. graphtools: tools for building and manipulating graphs in Python. Software, PyPI 'graphtools' (version studied: commit 785c0c1, 2025-10-27).
  [Code](https://github.com/KrishnaswamyLab/graphtools) · [Docs](https://graphtools.readthedocs.io/)
  Used in: `nds.graphs.adaptive_gaussian_kernel`, `nds.graphs.neighbour_bandwidths`, `nds.graphs.apply_anisotropy`, `configs/reference_graphtools_standard.yaml`, `tests/test_graphs_and_spectra.py::test_alpha_decay_kernel_matches_the_graphtools_formula`
- **PHATE**: Moon, K. R., van Dijk, D., Wang, Z., Gigante, S., Burkhardt, D. B., Chen, W. S., Yim, K., van den Elzen, A., Hirn, M. J., Coifman, R. R., Ivanova, N. B., Wolf, G., & Krishnaswamy, S. (2019). Visualizing structure and transitions in high-dimensional biological data. Nature Biotechnology, 37(12), 1482-1492.
  [DOI](https://doi.org/10.1038/s41587-019-0336-3) · [bioRxiv](https://www.biorxiv.org/content/10.1101/120378v4) · [PMC (open copy)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7073148) · [Code](https://github.com/KrishnaswamyLab/PHATE) · [Docs](https://phate.readthedocs.io/) · [Colab (embryoid body)](https://colab.research.google.com/github/KrishnaswamyLab/PHATE/blob/main/Python/tutorial/EmbryoidBody.ipynb) · [Colab (tree)](https://colab.research.google.com/github/KrishnaswamyLab/PHATE/blob/main/Python/tutorial/PHATE_tree.ipynb)
  Used in: `nds.spectra.diffusion_spectral_entropy`, `configs/reference_graphtools_standard.yaml`, `docs/method-notes.md §4`
  (verification: partly; see the research files)
- **M-PHATE: a graph over hidden units**: Gigante, S., Charles, A. S., Krishnaswamy, S., & Mishne, G. (2019). Visualizing the PHATE of neural networks. Advances in Neural Information Processing Systems 32 (NeurIPS 2019).
  [Paper (NeurIPS PDF)](https://papers.nips.cc/paper_files/paper/2019/file/b4d168b48157c623fbd095b4a565b5bb-Paper.pdf) · [NeurIPS poster page](https://neurips.cc/virtual/2019/poster/13391) · [Video abstract](https://www.youtube.com/watch?v=P9jacujC1Bc) · [Code](https://github.com/scottgigante/m-phate)
  Used in: `nds.moments.NeuronMoments`, `nds.graphs.find_dead_neurons`, `nds.graphs.correlation_distance_squared`, `nds.graphs.neuron_graph_from_covariance`
- **MM-PHATE: training phases from entropy**: Xie, J., Kohler Voinov, L. C., Mudrik, N., Mishne, G., & Charles, A. S. (2026). Multiway Multislice PHATE: Visualizing hidden dynamics of RNNs through training. Transactions on Machine Learning Research (TMLR).
  [arXiv PDF](https://arxiv.org/pdf/2406.01969) · [OpenReview](https://openreview.net/forum?id=9Yr4V7iZsq) · [Code](https://github.com/JianchengXie/MMPHATE_REPRO)
  Used in: `docs/method-notes.md §4b`, `nds.spectra.subspace_localization`, `nds.controls.rotate_live_covariance`
- **Diffusion spectral entropy (DSE)**: Liao, D., Liu, C., Christensen, B. W., Tong, A., Huguet, G., Wolf, G., Nickel, M., Adelstein, I., & Krishnaswamy, S. (2023). Assessing neural network representations during training using noise-resilient diffusion spectral entropy. ICML 2023 Workshop on Topology, Algebra, and Geometry in Machine Learning (TAG-ML). The authors also list IEEE CISS 2024.
  [arXiv PDF](https://arxiv.org/pdf/2312.04823) · [arXiv HTML](https://arxiv.org/html/2312.04823v1) · [Code (DSE functions only)](https://github.com/ChenLiu-1996/DiffusionSpectralEntropy) · [Slides](https://chenliu-1996.github.io/slides/DSE_slides.pdf)
  Used in: `nds.spectra.diffusion_spectral_entropy`, `nds.graphs.fixed_bandwidth_gaussian_kernel`, `nds.analysis.spectrum_numbers`, `tests/test_graphs_and_spectra.py::test_dse_matches_liao_et_al_formula`, `docs/method-notes.md §4`, `docs/method-notes.md §4b`
  (verification: partly; see the research files)
- **Diffusion condensation**: Brugnone, N., Gonopolskiy, A., Moyle, M. W., Kuchroo, M., van Dijk, D., Moon, K. R., Colón-Ramos, D., Wolf, G., Hirn, M. J., & Krishnaswamy, S. (2019). Coarse graining of data via inhomogeneous diffusion condensation. 2019 IEEE International Conference on Big Data (Big Data), 2624-2633.
  [arXiv PDF](https://arxiv.org/pdf/1907.04463) · [IEEE Xplore](https://ieeexplore.ieee.org/document/9006013) · [Code (diffusion_condensation)](https://github.com/KrishnaswamyLab/diffusion_condensation) · [Code (CATCH)](https://github.com/KrishnaswamyLab/CATCH)
  (verification: partly; see the research files)
- **Theory of diffusion condensation**: Huguet, G., Tong, A., Rieck, B., Huang, J., Kuchroo, M., Hirn, M., Wolf, G., & Krishnaswamy, S. (2023). Time-inhomogeneous diffusion geometry and topology. SIAM Journal on Mathematics of Data Science, 5(2), 346-372.
  [arXiv PDF](https://arxiv.org/pdf/2203.14860) · [NSF PAR copy](https://par.nsf.gov/servlets/purl/10434231) · [DOI](https://doi.org/10.1137/21M1462945)
- **Multiscale PHATE**: Kuchroo, M., Huang, J., Wong, P., Grenier, J.-C., Shung, D., Tong, A., Lucas, C., Klein, J., Burkhardt, D. B., Gigante, S., ..., Hirn, M. J., Wilson, F. P., Hussin, J. G., Wolf, G., Iwasaki, A., & Krishnaswamy, S. (2022). Multiscale PHATE identifies multimodal signatures of COVID-19. Nature Biotechnology, 40(5), 681-691.
  [DOI](https://doi.org/10.1038/s41587-021-01186-x) · [bioRxiv PDF](https://www.biorxiv.org/content/10.1101/2020.11.15.383661.full.pdf) · [Code](https://github.com/KrishnaswamyLab/Multiscale_PHATE)
  (verification: partly; see the research files)
- **HeatGeo: heat-kernel geodesics**: Huguet, G., Tong, A., De Brouwer, E., Zhang, Y., Wolf, G., Adelstein, I., & Krishnaswamy, S. (2023). A heat diffusion perspective on geodesic preserving dimensionality reduction. Advances in Neural Information Processing Systems 36 (NeurIPS 2023).
  [arXiv PDF](https://arxiv.org/pdf/2305.19043) · [OpenReview](https://openreview.net/forum?id=HNd4qTJxkW) · [Supplement](https://papers.neurips.cc/paper_files/paper/2023/file/16063a1c0f0cddd4894585cf44cebb2c-Supplemental-Conference.pdf) · [Code](https://github.com/KrishnaswamyLab/HeatGeo) · [Colab (swiss roll)](https://colab.research.google.com/drive/1SkxGfWz8KLQpcea9MsdP7QEoe-yh5Iai?usp=sharing)
  Used in: `nds.spectra.diffusion_spectrum`, `docs/method-notes.md §4b`
- **Diffusion EMD**: Tong, A. Y., Huguet, G., Natik, A., Macdonald, K., Kuchroo, M., Coifman, R., Wolf, G., & Krishnaswamy, S. (2021). Diffusion Earth Mover's Distance and distribution embeddings. Proceedings of the 38th International Conference on Machine Learning (ICML 2021), PMLR 139, 10336-10346.
  [PMLR PDF](http://proceedings.mlr.press/v139/tong21a/tong21a.pdf) · [Code](https://github.com/KrishnaswamyLab/DiffusionEMD)
  (verification: partly; see the research files)
- **RNN representation dynamics**: Horoi, S., Lajoie, G., & Wolf, G. (2020). Internal representation dynamics and geometry in recurrent neural networks. arXiv:2001.03255 (preprint; no venue found).
  [arXiv PDF](https://arxiv.org/pdf/2001.03255)
  (verification: partly; see the research files)
- **PHATE and topology on loss landscapes**: Horoi, S., Huang, J., Rieck, B., Lajoie, G., Wolf, G., & Krishnaswamy, S. (2022). Exploring the geometry and topology of neural network loss landscapes. In Advances in Intelligent Data Analysis XX (IDA 2022).
  [arXiv PDF](https://arxiv.org/pdf/2102.00485) · [Code](https://github.com/KrishnaswamyLab/NN-LossLandscape-PHATE)
  (verification: partly; see the research files)
- **When CKA can be fooled**: Davari, M., Horoi, S., Natik, A., Lajoie, G., Wolf, G., & Belilovsky, E. (2023). Reliability of CKA as a similarity measure in deep learning. International Conference on Learning Representations (ICLR 2023), poster.
  [arXiv PDF](https://arxiv.org/pdf/2210.16156) · [OpenReview](https://openreview.net/forum?id=8HRvyxc606)
  (verification: partly; see the research files)
- **T-PHATE: trajectories through time**: Busch, E. L., Huang, J., Benz, A., Wallenstein, T., Lajoie, G., Wolf, G., Krishnaswamy, S., & Turk-Browne, N. B. (2023). Multi-view manifold learning of human brain-state trajectories. Nature Computational Science, 3(3), 240-253.
  [DOI](https://doi.org/10.1038/s43588-023-00419-0) · [PMC (open copy)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10487346) · [Code](https://github.com/KrishnaswamyLab/TPHATE) · [Analysis capsule](https://github.com/ericabusch/tphate_analysis_capsule)
  (verification: partly; see the research files)

## III. How LLM representations change during pretraining

- **RankMe and alpha-ReQ across Pythia checkpoints**: Melody Zixuan Li, Kumar Krishna Agrawal, Arna Ghosh, Komal Kumar Teru, Adam Santoro, Guillaume Lajoie, Blake A. Richards (2025). Tracing the Representation Geometry of Language Models from Pretraining to Post-training. NeurIPS 2025.
  [arXiv](https://arxiv.org/abs/2509.23024) · [PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/4eac651e6c06d6b374332cd14ab50f55-Paper-Conference.pdf) · [Code](https://github.com/melodylizx/Tracing-the-representation-geometry-of-language-models) · [Project page](https://melodylizx.github.io/llm-geometry-project/)
  Used in: `nds.baselines.rankme_li`, `nds.baselines.alpha_req`, `nds.baselines.baseline_summary`, `nds.activations.load_pythia`, `docs/method-notes.md §5`, `DECISIONS.md D7`
- **RankMe effective rank**: Quentin Garrido, Randall Balestriero, Laurent Najman, Yann LeCun (2023). RankMe: Assessing the Downstream Performance of Pretrained Self-Supervised Representations by Their Rank. ICML 2023, PMLR 202.
  [arXiv](https://arxiv.org/abs/2210.02885) · [PDF](https://arxiv.org/pdf/2210.02885)
  Used in: `nds.baselines.rankme_garrido`, `nds.baselines.rankme_from_singular_values`, `nds.baselines.rankme_li`, `docs/method-notes.md §5`
- **alpha-ReQ spectral decay exponent**: Kumar Krishna Agrawal, Arnab Kumar Mondal, Arna Ghosh, Blake A. Richards (2022). α-ReQ: Assessing Representation Quality in Self-Supervised Learning by measuring eigenspectrum decay. NeurIPS 2022.
  [PDF](https://papers.nips.cc/paper_files/paper/2022/file/70596d70542c51c8d9b4e423f4bf2736-Paper-Conference.pdf) · [Code](https://github.com/kumarkrishna/fastssl) · [Blog](https://people.eecs.berkeley.edu/~krishna/blog/nn/ssl.html) · [Mila article](https://mila.quebec/en/article/a-req/)
  Used in: `nds.baselines.alpha_req`, `docs/method-notes.md §5`, `DECISIONS.md D7`
  (verification: partly; see the research files)
- **Prompt entropy across layers and training**: Oscar Skean, Md Rifat Arefin, Dan Zhao, Niket Patel, Jalal Naghiyev, Yann LeCun, Ravid Shwartz-Ziv (2025). Layer by Layer: Uncovering Hidden Representations in Language Models. ICML 2025.
  [arXiv](https://arxiv.org/abs/2502.02013) · [PDF](https://arxiv.org/pdf/2502.02013) · [Code](https://github.com/OFSkean/information_flow)
  Used in: `nds.baselines.prompt_entropy`, `nds.baselines.prompt_entropy_from_gram`, `nds.baselines.normalised_token_rows`, `nds.baselines.matrix_entropy`, `nds.activations.collect_neuron_moments`, `DECISIONS.md D7`
  (verification: partly; see the research files)
- **Intrinsic dimension transition near step 1000**: Jin Hwa Lee, Thomas Jiralerspong, Lei Yu, Yoshua Bengio, Emily Cheng (2025). Geometric Signatures of Compositionality Across a Language Model's Lifetime. Proc. 63rd ACL (Vol. 1: Long Papers), pp. 5292-5320.
  [arXiv](https://arxiv.org/abs/2410.01444) · [PDF](https://aclanthology.org/2025.acl-long.265.pdf) · [DOI](https://doi.org/10.18653/v1/2025.acl-long.265) · [Code](https://github.com/jinhl9/llm-compositionality-lifetime)
  Used in: `nds.checkpoints.week_one_steps`, `docs/method-notes.md §7`
- **Mid-layer intrinsic-dimension peak**: Emily Cheng, Diego Doimo, Corentin Kervadec, Iuri Macocco, Jade Yu, Alessandro Laio, Marco Baroni (2025). Emergence of a High-Dimensional Abstraction Phase in Language Transformers. ICLR 2025.
  [arXiv](https://arxiv.org/abs/2405.15471) · [PDF](https://arxiv.org/pdf/2405.15471) · [Code](https://github.com/chengemily1/id-llm-abstraction)
- **Intrinsic-dimension profile across layers**: Lucrezia Valeriani, Diego Doimo, Francesca Cuturello, Alessandro Laio, Alessio Ansuini, Alberto Cazzaniga (2023). The geometry of hidden representations of large transformer models. NeurIPS 2023.
  [arXiv](https://arxiv.org/abs/2302.00294) · [PDF](https://arxiv.org/pdf/2302.00294) · [Code](https://github.com/diegodoimo/geometry_representations)
- **Pythia checkpoint suite**: Stella Biderman, Hailey Schoelkopf, Quentin Anthony, Herbie Bradley, Kyle O'Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, Aviya Skowron, Lintang Sutawika, Oskar van der Wal (2023). Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling. ICML 2023, PMLR 202:2397-2430.
  [arXiv](https://arxiv.org/abs/2304.01373) · [PDF](https://arxiv.org/pdf/2304.01373) · [Code](https://github.com/EleutherAI/pythia) · [Models](https://huggingface.co/EleutherAI)
  Used in: `nds.checkpoints.all_pythia_steps`, `nds.checkpoints.revision_name`, `nds.checkpoints.tokens_seen`, `nds.checkpoints.week_one_steps`, `nds.activations.load_pythia`, `nds.probe_text.build_probe_set`, `docs/method-notes.md §6`
  (verification: partly; see the research files)
- **PolyPythias extra seeds**: Oskar van der Wal, Pietro Lesci, Max Müller-Eberstein, Naomi Saphra, Hailey Schoelkopf, Willem Zuidema, Stella Biderman (2025). PolyPythias: Stability and Outliers across Fifty Language Model Pre-Training Runs. ICLR 2025.
  [arXiv](https://arxiv.org/abs/2503.09543) · [PDF](https://arxiv.org/pdf/2503.09543) · [OpenReview](https://openreview.net/forum?id=bmrYu2Ekdz) · [Code](https://github.com/EleutherAI/pythia/tree/main/polypythias) · [Models](https://huggingface.co/collections/EleutherAI/polypythias) · [Evals](https://huggingface.co/datasets/EleutherAI/polypythias-evals)
  Used in: `docs/method-notes.md §6`
  (verification: partly; see the research files)
- **TransformerLens hooks library**: Neel Nanda and Joseph Bloom (2022). TransformerLens. Maintained by Bryce Meyer and Jonah Larson. https://github.com/TransformerLensOrg/TransformerLens.
  [Code](https://github.com/TransformerLensOrg/TransformerLens) · [Colab (induction heads)](https://colab.research.google.com/github/ckkissane/induction-heads-transformer-lens/blob/main/Induction_Heads_Phase_Change.ipynb)
  Used in: `nds.activations.collect_neuron_moments`, `nds.induction.induction_scores`, `docs/method-notes.md §7`, `DECISIONS.md D3`
  (verification: partly; see the research files)
- **nnsight tracing library**: NDIF team. nnsight (v0.8.0; clone of 2026-10-16). https://github.com/ndif-team/nnsight.
  [Code](https://github.com/ndif-team/nnsight) · [Colab](https://colab.research.google.com/github/ndif-team/nnsight/blob/main/NNsight_Walkthrough.ipynb)
  Used in: `docs/method-notes.md §7`
  (verification: partly; see the research files)
- **Graph probing of correlation networks**: Yu Zheng, Yuan Yuan, Yue Zhuo, Yong Li, Gabriel Kreiman, Tomaso Poggio, Paolo Santi (2025). Probing Neural Topology of Large Language Models. arXiv preprint.
  [arXiv](https://arxiv.org/abs/2506.01042) · [PDF](https://arxiv.org/pdf/2506.01042) · [Code](https://github.com/DavyMorgan/llm-graph-probing)
  Used in: `nds.graphs.correlation_from_covariance`, `nds.graphs.correlation_distance_squared`, `nds.graphs.adaptive_gaussian_kernel`, `nds.graphs.neuron_graph_from_covariance`
  (verification: partly; see the research files)
- **Weight-graph clusterability**: Satvik Golechha, Maheep Chaudhary, Joan Velja, Alessandro Abate, Nandi Schoots (2025). Studying Cross-cluster Modularity in Neural Networks. arXiv preprint.
  [arXiv](https://arxiv.org/abs/2502.02470) · [PDF](https://arxiv.org/pdf/2502.02470)
  Used in: `nds.spectra.eigengap_module_count`, `nds.modules.spectral_partition`
- **Neuron experts emerge in pretraining**: Zhengyan Zhang, Zhiyuan Zeng, Yankai Lin, Chaojun Xiao, Xiaozhi Wang, Xu Han, Zhiyuan Liu, Ruobing Xie, Maosong Sun, Jie Zhou (2023). Emergent Modularity in Pre-trained Transformers. Findings of ACL 2023.
  [arXiv](https://arxiv.org/abs/2305.18390) · [Code](https://github.com/THUNLP/modularity-analysis)
  Used in: `nds.modules.spectral_partition`, `nds.flow.match_modes`
  (verification: partly; see the research files)
- **Component graphs over Pythia-14M training**: Elisabetta Rocchetti (2025). Modeling Transformers as complex networks to analyze learning dynamics. arXiv preprint.
  [arXiv](https://arxiv.org/abs/2509.15269) · [PDF](https://arxiv.org/pdf/2509.15269)
- **Feature co-activation connectome**: Ruixuan Deng, Zehao Jin (2026). The Connectome of a Large Language Model. Poster, Workshop on Mechanistic Interpretability, ICML 2026.
  [ICML](https://icml.cc/virtual/2026/79277) · [Workshop](https://icml.cc/virtual/2026/workshop/54071)
  (verification: partly; see the research files)
- **Spectral early warnings of transitions**: Max Hennick, Guillaume Corlouer (2026). From Density Matrices to Phase Transitions in Deep Learning: Spectral Early Warnings and Interpretability. arXiv preprint.
  [arXiv](https://arxiv.org/abs/2603.29805) · [PDF](https://arxiv.org/pdf/2603.29805)
  Used in: `nds.timeline.spectral_flow`, `nds.timeline.avoided_crossing_candidates`, `nds.flow.fit_avoided_crossing`
- **LLC finds developmental stages**: Jesse Hoogland, George Wang, Matthew Farrugia-Roberts, Liam Carroll, Susan Wei, Daniel Murfet (2025). Loss Landscape Degeneracy and Stagewise Development in Transformers. TMLR 2025.
  [arXiv](https://arxiv.org/abs/2402.02364) · [PDF](https://arxiv.org/pdf/2402.02364) · [Code](https://github.com/timaeus-research/devinterp) · [Docs](https://devinterp.timaeus.co/)
  (verification: partly; see the research files)
- **Per-head refined LLC**: George Wang, Jesse Hoogland, Stan van Wingerden, Zach Furman, Daniel Murfet (2025). Differentiation and Specialization of Attention Heads via the Refined Local Learning Coefficient. ICLR 2025.
  [arXiv](https://arxiv.org/abs/2410.02984) · [Code](https://github.com/timaeus-research/devinterp)
  (verification: partly; see the research files)
- **Susceptibility UMAP over training**: George Wang, Garrett Baker, Andrew Gordon, Daniel Murfet (2025). Embryology of a Language Model. arXiv preprint.
  [arXiv](https://arxiv.org/abs/2508.00331)
  Used in: `nds.timeline.spectral_flow`
  (verification: partly; see the research files)
- **Susceptibility 'spectroscopy' on Pythia-14M**: Andrew Gordon, Garrett Baker, George Wang, William Snell, Stan van Wingerden, Daniel Murfet (2026). Towards Spectroscopy: Susceptibility Clusters in Language Models. arXiv preprint.
  [arXiv](https://arxiv.org/abs/2601.12703)
  Used in: `nds.graphs.adaptive_gaussian_kernel`, `nds.modules.conductances`
  (verification: partly; see the research files)
- **Universal neurons across seeds**: Wes Gurnee, Theo Horsley, Zifan Carl Guo, Tara Rezaei Kheirkhah, Qinyi Sun, Will Hathaway, Neel Nanda, Dimitris Bertsimas (2024). Universal Neurons in GPT2 Language Models. TMLR 2024.
  [arXiv](https://arxiv.org/abs/2401.12181) · [PDF](https://arxiv.org/pdf/2401.12181) · [Code](https://github.com/wesg52/universal-neurons)
  Used in: `nds.moments.NeuronMoments`, `nds.moments.BlockedNeuronMoments`, `nds.activations.collect_neuron_moments`, `nds.probe_text.build_probe_set`, `nds.controls.random_orthogonal_matrix`
- **Massive activations**: Mingjie Sun, Xinlei Chen, J. Zico Kolter, Zhuang Liu (2024). Massive Activations in Large Language Models. First Conference on Language Modeling (COLM 2024); also an oral at the ICLR 2024 ME-FoMo workshop. arXiv 2402.17762.
  [arXiv](https://arxiv.org/abs/2402.17762) · [Project page](https://eric-mingjie.github.io/massive-activations/index.html) · [Code](https://github.com/locuslab/massive-activations)
  Used in: `nds.analysis.find_massive_neurons`, `nds.activations.summarise_on_device`, `nds.probe_text.build_probe_set`, `DECISIONS.md D9`
  (verification: partly; see the research files)
- **Dead, n-gram and positional neurons**: Elena Voita, Javier Ferrando, Christoforos Nalmpantis (2024). Neurons in Large Language Models: Dead, N-gram, Positional. Findings of ACL 2024.
  [arXiv](https://arxiv.org/abs/2309.04827)
  Used in: `nds.graphs.find_dead_neurons`, `nds.controls.rotate_live_covariance`, `nds.moments.NeuronMoments.firing_rate`
  (verification: partly; see the research files)
- **Induction-head phase change**: Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, ..., Chris Olah (26 authors) (2022). In-context Learning and Induction Heads. Transformer Circuits Thread.
  [arXiv](https://arxiv.org/abs/2209.11895) · [HTML](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
  Used in: `nds.induction.induction_scores`, `nds.induction.repeated_random_tokens`

## IV. Circuits and naming the events

- **Sparse feature circuits (SVA)**: Marks, S., Rager, C., Michaud, E. J., Belinkov, Y., Bau, D., Mueller, A. (2025). Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models. ICLR 2025.
  [arXiv](https://arxiv.org/abs/2403.19647) · [PDF](https://arxiv.org/pdf/2403.19647) · [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/file/3ba4d47a83e498c2b1a0868cba20f6de-Paper-Conference.pdf) · [ICLR oral](https://iclr.cc/virtual/2025/oral/31874) · [Code](https://github.com/saprmarks/feature-circuits) · [Code (dictionaries)](https://github.com/saprmarks/dictionary_learning) · [Project page](https://features.baulab.info/) · [Clusters demo](https://feature-circuits.xyz/) · [Neuronpedia](https://www.neuronpedia.org/p70d-sm) · [Dictionaries (HF)](https://huggingface.co/saprmarks/pythia-70m-deduped-saes)
  Used in: `nds.neuron_attribution.neuron_attribution_patching`, `nds.neuron_attribution.neuron_integrated_gradients`, `nds.neuron_attribution.answer_logit_difference`, `nds.naming.feature_loadings_on_neurons`, `docs/method-notes.md §6`, `docs/method-notes.md §8`
- **Pythia-70M-deduped MLP-out dictionaries**: Marks, S. (with A. Mueller) (2023). Some open-source dictionaries and dictionary learning infrastructure. AI Alignment Forum / LessWrong post, 5 Dec 2023 (edited 7 Feb 2024).
  [Post](https://www.alignmentforum.org/posts/AaoWLcmpY3LKvtdyq/some-open-source-dictionaries-and-dictionary-learning) · [Code](https://github.com/saprmarks/dictionary_learning)
  Used in: `nds.naming.feature_loadings_on_neurons`, `nds.naming.mean_absolute_contributions`, `tests/test_naming.py`, `docs/method-notes.md §8`
- **Transcoders for MLP circuits**: Dunefsky, J., Chlenski, P., Nanda, N. (2024). Transcoders Find Interpretable LLM Feature Circuits. NeurIPS 2024.
  [arXiv](https://arxiv.org/abs/2406.11944) · [PDF](https://arxiv.org/pdf/2406.11944) · [Proceedings](https://papers.neurips.cc/paper_files/paper/2024/file/2b8f4db0464cc5b6e9d5e6bea4b9f308-Paper-Conference.pdf) · [Code](https://github.com/jacobdunefsky/transcoder_circuits) · [GPT-2 transcoders (HF)](https://huggingface.co/pchlenski/gpt2-transcoders)
  Used in: `nds.naming.feature_loadings_on_neurons`, `nds.naming.mean_absolute_contributions`
- **Transcoders vs SAEs (sparsify)**: Paulo, G., Shabalin, S., Belrose, N. (2025). Transcoders Beat Sparse Autoencoders for Interpretability. arXiv preprint.
  [arXiv](https://arxiv.org/abs/2501.18823) · [PDF](https://arxiv.org/pdf/2501.18823) · [Code](https://github.com/EleutherAI/sparsify) · [SAEs pythia-70m-deduped (HF)](https://huggingface.co/EleutherAI/sae-pythia-70m-deduped-32k) · [SAEs pythia-70m (HF)](https://huggingface.co/EleutherAI/sae-pythia-70m-32k)
- **Circuits over Pythia training**: Tigges, C., Hanna, M., Yu, Q., Biderman, S. (2024). LLM Circuit Analyses Are Consistent Across Training and Scale. NeurIPS 2024 (poster).
  [arXiv](https://arxiv.org/abs/2407.10827) · [PDF](https://arxiv.org/pdf/2407.10827v2) · [NeurIPS](https://neurips.cc/virtual/2024/poster/96762)
  Used in: `docs/method-notes.md §6`, `configs/pythia70m_all.yaml`
  (verification: partly; see the research files)
- **Crosscoders across checkpoints**: Bayazit, D., Mueller, A., Bosselut, A. (2025). Crosscoding Through Time: Tracking Emergence & Consolidation of Linguistic Representations Throughout LLM Pretraining. arXiv:2509.05291 (v2 2026). Venue not confirmed.
  [arXiv](https://arxiv.org/abs/2509.05291) · [PDF](https://arxiv.org/pdf/2509.05291v2) · [Code](https://github.com/bayazitdeniz/crosscoding-through-time)
  (verification: partly; see the research files)
- **SAE-Track: per-checkpoint SAEs**: Xu, Y., Wang, Y., Huang, H., Wang, H. (2024). Tracking the Feature Dynamics in LLM Training: A Mechanistic Study. arXiv preprint.
  [arXiv](https://arxiv.org/abs/2412.17626) · [PDF](https://arxiv.org/pdf/2412.17626) · [Code](https://github.com/Superposition09m/SAE-Track)
- **Attribution graphs and supernodes**: Ameisen, E., Lindsey, J., Pearce, A., Gurnee, W., Turner, N. L., et al. (2025). Circuit Tracing: Revealing Computational Graphs in Language Models. Transformer Circuits Thread, 27 Mar 2025.
  [Paper](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) · [Open implementation](https://github.com/decoderesearch/circuit-tracer)
- **Biology of Claude 3.5 Haiku**: Lindsey, J., Gurnee, W., Ameisen, E., Chen, B., Pearce, A., et al. (2025). On the Biology of a Large Language Model. Transformer Circuits Thread, 27 Mar 2025.
  [Paper](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)
- **circuit-tracer library**: Hanna, M., Piotrowski, M., Lindsey, J., Ameisen, E. (2025). Circuit-Tracer: A New Library for Finding Feature Circuits. Proceedings of the 8th BlackboxNLP Workshop, pp. 239-249, Suzhou.
  [PDF (preview)](https://preview.aclanthology.org/ingest-emnlp/2025.blackboxnlp-1.14.pdf) · [DOI](https://doi.org/10.18653/v1/2025.blackboxnlp-1.14) · [Code](https://github.com/decoderesearch/circuit-tracer) · [Tutorial notebook](https://github.com/decoderesearch/circuit-tracer/blob/main/demos/circuit_tracing_tutorial.ipynb)
  Used in: `nds.naming.smallest_set_holding_share`
  (verification: partly; see the research files)
- **ADAG: automatic neuron supernodes**: Arora, A., Wu, Z., Steinhardt, J., Schwettmann, S. (2026). ADAG: Automatically Describing Attribution Graphs. arXiv preprint. Companion: Arora, Wu, Steinhardt, Schwettmann (2026). Language Model Circuits Are Sparse in the Neuron Basis. arXiv:2601.22594.
  [arXiv](https://arxiv.org/abs/2604.07615) · [PDF](https://arxiv.org/pdf/2604.07615) · [Companion (arXiv)](https://arxiv.org/abs/2601.22594) · [Code](https://github.com/TransluceAI/circuits)
  Used in: `nds.neuron_attribution.neuron_attribution_patching`, `nds.naming.smallest_set_holding_share`, `nds.naming.retention_probability`, `nds.naming.retention_z_score`, `nds.modules.spectral_partition`
- **ACDC: automated circuit discovery**: Conmy, A., Mavor-Parker, A. N., Lynch, A., Heimersheim, S., Garriga-Alonso, A. (2023). Towards Automated Circuit Discovery for Mechanistic Interpretability. NeurIPS 2023 (spotlight per README).
  [arXiv](https://arxiv.org/abs/2304.14997) · [PDF](https://arxiv.org/pdf/2304.14997v4) · [NeurIPS](https://neurips.cc/virtual/2023/poster/72666) · [Code](https://github.com/ArthurConmy/Automatic-Circuit-Discovery) · [Colab](https://colab.research.google.com/github/ArthurConmy/Automatic-Circuit-Discovery/blob/main/notebooks/colabs/ACDC_Main_Demo.ipynb)
- **Edge attribution patching (EAP)**: Syed, A., Rager, C., Conmy, A. (2024). Attribution Patching Outperforms Automated Circuit Discovery. Proceedings of the 7th BlackboxNLP Workshop, pp. 407-416. (arXiv 2310.10348, 2023; earlier NeurIPS 2023 ATTRIB workshop.)
  [arXiv](https://arxiv.org/abs/2310.10348) · [DOI](https://doi.org/10.18653/v1/2024.blackboxnlp-1.25) · [Code (original)](https://github.com/Aaquib111/edge-attribution-patching) · [Code (EAP-IG)](https://github.com/hannamw/EAP-IG) · [Code (faithfulness, COLM 2024)](https://github.com/hannamw/eap-ig-faithfulness)
  Used in: `nds.neuron_attribution.neuron_attribution_patching`, `nds.neuron_attribution.neuron_integrated_gradients`, `docs/method-notes.md §7`
- **SVA templates and causal mediation**: Finlayson, M., Mueller, A., Gehrmann, S., Shieber, S., Linzen, T., Belinkov, Y. (2021). Causal Analysis of Syntactic Agreement Mechanisms in Neural Language Models. Proceedings of ACL-IJCNLP 2021 (Vol. 1: Long Papers), pp. 1828-1843.
  [arXiv](https://arxiv.org/abs/2106.06087) · [PDF](https://aclanthology.org/2021.acl-long.144.pdf) · [DOI](https://doi.org/10.18653/v1/2021.acl-long.144) · [Code](https://github.com/mattf1n/lm-intervention)
- **Neuronpedia feature browser**: Neuronpedia. Model pythia-70m-deduped, release p70d-sm, 'Sparse Autoencoders for Pythia-70M-Deduped' (April 2024).
  [Release p70d-sm](https://www.neuronpedia.org/p70d-sm) · [Layer 3 MLP source](https://www.neuronpedia.org/pythia-70m-deduped/3-mlp-sm)
  (verification: partly; see the research files)
- **SAELens SAE library and registry**: Bloom, J., Tigges, C., Duong, A., Chanin, D. (maintainers). SAELens, v6.54.5. GitHub: decoderesearch/SAELens.
  [Code](https://github.com/decoderesearch/SAELens)
  (verification: partly; see the research files)
- **TransformerLens hooked models**: TransformerLensOrg. TransformerLens. GitHub: TransformerLensOrg/TransformerLens.
  [Code](https://github.com/TransformerLensOrg/TransformerLens)
  Used in: `docs/method-notes.md §7`, `nds.activations.collect_neuron_moments`
  (verification: partly; see the research files)
- **nnsight tracing library**: Fiotto-Kaufman, J., Loftus, A. R., Todd, E., Brinkmann, J., Pal, K., Troitskii, D., Ripa, M., Belfki, A., Rager, C., Juang, C., Mueller, A., Marks, S., Sen Sharma, A., Lucchetti, F., Prakash, N., Brodley, C., Guha, A., Bell, J., Wallace, B., Bau, D. (2025). NNsight and NDIF: Democratizing Access to Open-Weight Foundation Model Internals. ICLR 2025.
  [arXiv](https://arxiv.org/abs/2407.14561) · [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e6c65eb9b56719c1aa45ff73874de317-Abstract-Conference.html) · [ICLR poster](https://iclr.cc/virtual/2025/poster/29909) · [Project](https://nnsight.net/) · [NDIF](https://ndif.baulab.info/) · [Code](https://github.com/ndif-team/nnsight)
  Used in: `docs/method-notes.md §7`
  (verification: partly; see the research files)

## V. Round 2: sources added 22 Oct 2026

- **Effective number of tokens**: Bartlett, M. S. (1946). On the theoretical specification and sampling properties of autocorrelated time-series. Supplement to the Journal of the Royal Statistical Society 8(1).
  Used in: `nds.noise.effective_token_count`, `nds.noise.lag_one_correlations`, `nds.activations.median_lag_one_correlation`, `nds.activations.collect_neuron_moments`, `scripts/collect.py`, `tests/test_token_sample_and_noise.py::test_effective_tokens_and_lag_one_correlation`, `tests/test_token_sample_and_noise.py::test_collection_lag_sums_match_the_direct_formula`
  (verification: no; see the research files)
- **Marchenko-Pastur noise band**: Marchenko, V. A., Pastur, L. A. (1967). Distribution of eigenvalues for some sets of random matrices. Mathematics of the USSR-Sbornik 1(4).
  [Notebook: Marchenko-Pastur plots (WeightWatcher)](https://github.com/CalculatedContent/WeightWatcher/blob/master/examples/Marchenko_Pastur_Plots.ipynb) · [arXiv (Aarts et al. 2024, nano-GPT weights)](https://arxiv.org/abs/2411.13512)
  Used in: `nds.noise.marchenko_pastur_edges`, `tests/test_token_sample_and_noise.py::test_wishart_null_fills_the_marchenko_pastur_interval`
  (verification: no; see the research files)
- **BBP detectability threshold**: Baik, J., Ben Arous, G., Peche, S. (2005). Phase transition of the largest eigenvalue for nonnull complex sample covariance matrices. Annals of Probability 33(5).
  [DOI (Nadakuditi & Newman 2012, graph analogue)](https://doi.org/10.1103/PhysRevLett.108.188701) · [arXiv (Nadakuditi & Newman 2012)](https://arxiv.org/abs/1205.1813)
  Used in: `nds.noise`, `nds.noise.marchenko_pastur_edges`
  (verification: no; see the research files)
- **Bartlett decomposition (Wishart null)**: Bartlett, M. S. (1933). On the theory of statistical regression. Proceedings of the Royal Society of Edinburgh 53.
  Used in: `nds.noise.wishart_null_correlation`, `nds.analysis.analyse_layer`, `nds.token_sample.shuffle_neurons_independently`, `DECISIONS.md D36`, `tests/test_token_sample_and_noise.py::test_wishart_null_fills_the_marchenko_pastur_interval`, `tests/test_token_sample_and_noise.py::test_shuffle_keeps_each_neuron_and_kills_correlation`
  (verification: no; see the research files)
- **Level-spacing ratio <r>**: Oganesyan, V., Huse, D. A. (2007). Localization of interacting fermions at high temperature. Physical Review B 75, 155111. Atas, Y. Y., Bogomolny, E., Giraud, O., Roux, G. (2013). Distribution of the ratio of consecutive level spacings in random matrix ensembles. Physical Review Letters 110(8), 084101.
  [DOI (Oganesyan & Huse)](https://doi.org/10.1103/PhysRevB.75.155111) · [arXiv (Oganesyan & Huse)](https://arxiv.org/abs/cond-mat/0610854) · [arXiv (Atas et al.)](https://arxiv.org/abs/1212.5611) · [arXiv (Giraud et al. 2022, independent blocks)](https://arxiv.org/abs/2008.11173)
  Used in: `nds.spectra.spacing_ratio`, `nds.spectra.spectrum_summary`, `nds.analysis.short_summary`, `scripts/compare_runs.py`, `tests/test_level_statistics_and_subspaces.py::test_spacing_ratio_of_independent_levels_is_the_poisson_value`, `tests/test_level_statistics_and_subspaces.py::test_spacing_ratio_of_a_random_symmetric_matrix_is_the_goe_value`
  (verification: partly; see the research files)
- **Degree localisation confound**: Hata, S., Nakao, H. (2017). Localization of Laplacian eigenvectors on random networks. Scientific Reports 7. DOI 10.1038/s41598-017-01010-0.
  [DOI](https://doi.org/10.1038/s41598-017-01010-0) · [DOI (Pastor-Satorras & Castellano 2016, types of localisation)](https://doi.org/10.1038/srep18847)
  Used in: `nds.spectra.spectrum_summary`, `nds.analysis.short_summary`, `nds.analysis.layer_summary`
- **Adjusted Rand index (ARI)**: Hubert, L., Arabie, P. (1985). Comparing partitions. Journal of Classification 2(1).
  Used in: `nds.modules.adjusted_rand_index`, `nds.timeline.partition_stability`, `nds.timeline.partition_stability_series`, `scripts/analyse.py`, `scripts/compare_runs.py`, `DECISIONS.md D33`, `tests/test_token_sample_and_noise.py::test_adjusted_rand_index_extremes`, `tests/test_token_sample_and_noise.py::test_partition_stability_uses_common_neurons_only`
  (verification: no; see the research files)
- **Principal angles between subspaces**: Bjorck, A., Golub, G. H. (1973). Numerical methods for computing angles between linear subspaces. Mathematics of Computation 27(123).
  Used in: `nds.timeline.subspace_distance`, `nds.timeline.subspace_change_series`, `nds.flow.group_near_degenerate_modes`, `nds.flow.match_mode_groups`, `scripts/analyse.py`, `scripts/compare_runs.py`, `tests/test_level_statistics_and_subspaces.py::test_subspace_distance_extremes_and_mixing`, `tests/test_flow.py::test_group_matching_ignores_mixing_inside_a_degenerate_pair`
  (verification: no; see the research files)
- **Readout reorganisation, circular-shift null**: He, M., Shen, W. F., Iacob, A., Jovanovic, A., Qiu, X., Lane, N. D. (2026). Learning to Read Out: Unembedding Dynamics in Language Model Pretraining. arXiv:2610.21025. (The repository README says NeurIPS 2026; not confirmed on a NeurIPS page.)
  [arXiv](https://arxiv.org/abs/2610.21025) · [Code](https://github.com/hematteo/learning-to-read-out)
  Used in: `nds.timeline.circular_shift_p_value`, `nds.timeline.window_enrichment`, `nds.known_events.transitions_in_window`, `nds.known_events.KNOWN_EVENTS`, `scripts/analyse.py`, `tests/test_level_statistics_and_subspaces.py::test_circular_shift_null_finds_a_planted_burst_and_not_noise`, `tests/test_level_statistics_and_subspaces.py::test_known_events_and_their_windows`
  (verification: partly; see the research files)
- **Network density matrix and entropy**: De Domenico, M., Biamonte, J. (2016). Spectral entropies as information-theoretic tools for complex network comparison. Physical Review X 6, 041062.
  [arXiv](https://arxiv.org/abs/1609.01214)
  Used in: `nds.spectra.heat_capacity_curve`, `nds.spectra.spectrum_summary`, `tests/test_level_statistics_and_subspaces.py::test_heat_capacity_peaks_higher_with_modules_than_without`
- **Heat-capacity peak as a scale**: Villegas, P., Gabrielli, A., Santucci, F., Caldarelli, G., Gili, T. (2022). Laplacian paths in complex networks: information core emerges from entropic transitions. Physical Review Research 4, 033196.
  [DOI (Villegas et al. 2023, Laplacian renormalization group, Nature Physics)](https://doi.org/10.1038/s41567-022-01866-8) · [arXiv (Villegas et al. 2023)](https://arxiv.org/abs/2203.07230)
  Used in: `nds.spectra.heat_capacity_curve`, `nds.spectra.spectrum_summary`, `nds.analysis.short_summary`, `tests/test_level_statistics_and_subspaces.py::test_heat_capacity_of_two_levels_matches_the_schottky_formula`, `tests/test_level_statistics_and_subspaces.py::test_heat_capacity_peaks_higher_with_modules_than_without`
  (verification: partly; see the research files)
- **Knowledge entropy of FFN neurons**: Kim, J., Lee, H., Cho, H., Jang, J., Hwang, H., Won, S., Ahn, Y., Lee, D., Seo, M. (2025). Knowledge Entropy Decay during Language Model Pretraining Hinders New Knowledge Acquisition. ICLR 2025 (oral).
  [arXiv](https://arxiv.org/abs/2410.01380) · [Code](https://github.com/kaistAI/Knowledge-Entropy)
  Used in: `nds.baselines.knowledge_entropy`, `nds.baselines.knowledge_entropy_of_sample`, `nds.token_sample.token_sample_baselines`, `nds.activations.collect_neuron_moments`, `scripts/collect.py`, `scripts/analyse.py`, `tests/test_level_statistics_and_subspaces.py::test_knowledge_entropy_extremes_and_rotation_changes_it`
- **Embedding condensation (mean cosine)**: Liu, C., Sun, X., Xiao, X., Van Tassel, A., Xu, K., Reimann, K., Liao, D., Gerstein, M., Wang, T., Wang, X., Krishnaswamy, S. (2026). Dispersion Loss Counteracts Embedding Condensation and Improves Generalization in Small Language Models. ICML 2026 (poster).
  [arXiv](https://arxiv.org/abs/2602.00217) · [OpenReview](https://openreview.net/forum?id=pd6A7jB5D6) · [ICML 2026 poster page](https://icml.cc/virtual/2026/poster/61492) · [Project page](https://chenliu-1996.github.io/projects/LM-Dispersion/) · [Code](https://github.com/ChenLiu-1996/LM-Dispersion)
  Used in: `nds.token_sample.mean_token_cosine`, `nds.token_sample.token_sample_baselines`, `tests/test_token_sample_and_noise.py::test_mean_token_cosine_extremes_and_rotation`
- **TwoNN intrinsic dimension**: Facco, E., d'Errico, M., Rodriguez, A., Laio, A. (2017). Estimating the intrinsic dimension of datasets by a minimal neighborhood information. Scientific Reports 7, 12140.
  [Code (DADApy, compute_id_2NN)](https://github.com/sissa-data-science/DADApy) · [Notebook: intrinsic dimension (DADApy)](https://github.com/sissa-data-science/DADApy/blob/main/examples/notebook_on_intrinsic_dimension.ipynb)
  Used in: `nds.token_sample.twonn_intrinsic_dimension`, `nds.token_sample.token_sample_baselines`, `scripts/analyse.py`, `DECISIONS.md D35`, `tests/test_token_sample_and_noise.py::test_twonn_finds_the_dimension_of_a_flat_set`
  (verification: partly; see the research files)
- **HTSR alpha and WeightWatcher**: Martin, C. H., Mahoney, M. W. (2021). Implicit self-regularization in deep neural networks: evidence from random matrix theory and implications for learning. Journal of Machine Learning Research 22(165): 1-73. Software: WeightWatcher (CalculatedContent), v0.8.8.
  [Paper (JMLR)](https://jmlr.org/papers/v22/20-410.html) · [Code (WeightWatcher)](https://github.com/CalculatedContent/WeightWatcher) · [HTSR phases (weightwatcher.ai)](https://weightwatcher.ai/htsr.html) · [arXiv (Prakash & Martin 2025, HTSR over training)](https://arxiv.org/abs/2506.04434)
  Used in: `nds.baselines.htsr_alpha`, `scripts/collect.py`, `tests/test_baselines_and_controls.py::test_htsr_alpha_matches_one_plus_one_over_a_and_ignores_rotation`
- **Fitting power laws (MLE + KS)**: Clauset, A., Shalizi, C. R., Newman, M. E. J. (2009). Power-law distributions in empirical data. SIAM Review 51(4).
  Used in: `nds.baselines.htsr_alpha`
  (verification: no; see the research files)
- **NeuroMFA: closest prior work**: Xiao, X., Ping, H., Zhou, C., Cao, D., Li, Y., Zhou, Y.-Z., Li, S., Kanakaris, N., Bogdan, P. (2025). Neuron-based Multifractal Analysis of Neuron Interaction Dynamics in Large Models. ICLR 2025. (Earlier title: Exploring Neuron Interactions and Emergence in LLMs: From the Multifractal Analysis Perspective.)
  [arXiv](https://arxiv.org/abs/2402.09099) · [Code (anonymous mirror, not cloned)](https://anonymous.4open.science/r/Neuron_LLM-4DC8)
  (verification: partly; see the research files)
- **Crosscoder features over Pythia training**: Ge, X., Shu, W., Wu, J., Zhou, Y., He, Z., Qiu, X. (2026). Evolution of Concepts in Language Model Pre-Training. ICLR 2026 (per the arXiv comment). arXiv:2509.17196.
  [arXiv](https://arxiv.org/abs/2509.17196) · [Code (general SAE library, not cloned)](https://github.com/OpenMOSS/Language-Model-SAEs)
  Used in: `nds.known_events.KNOWN_EVENTS`, `nds.known_events.events_for_model`, `scripts/analyse.py`
- **SAE feature life histories**: Stecher, P., Radovanović, S., Sikimić, V., Kahle, R. (2026). Features have life history. And we should care. arXiv:2605.18789 (q-bio.NC).
  [arXiv](https://arxiv.org/abs/2605.18789)
  Used in: `nds.known_events.KNOWN_EVENTS`
- **Syntax neurons, a critical period**: Duan, X., Yao, Z., Zhang, Y., Wang, S., Cai, Z. G. (2025). How Syntax Specialization Emerges in Language Models. arXiv:2505.19548.
  [arXiv](https://arxiv.org/abs/2505.19548)
  Used in: `nds.known_events.KNOWN_EVENTS`, `DECISIONS.md D37`
- **Distilling Pythia diffusion geometry**: Steindl, Valdez Cordova, Bhaskar, Adelstein, Wolf, G., Krishnaswamy, S. (2026). Towards Distilling the Representational Geometry of Language Models. TAG-DS 2026 poster (extended abstract, non-archival), OpenReview Sp2IZrubi7.
  [OpenReview](https://openreview.net/forum?id=Sp2IZrubi7)
  (verification: partly; see the research files)
- **T-PHATE manifolds of learning brains**: Busch, E. L., Fincke, E. C., Lajoie, G., Krishnaswamy, S., Turk-Browne, N. B. (2026). Human learning of noninvasive brain-computer interfaces via manifold geometry. Nature Neuroscience. Preprint: Accelerated learning of a noninvasive human brain-computer interface via manifold geometry, bioRxiv 10.1101/2025.03.29.646109.
  [Article (link from Yale News)](https://www.nature.com/articles/s41593-026-02311-2) · [bioRxiv preprint](https://doi.org/10.1101/2025.03.29.646109) · [Data (Dryad)](https://doi.org/10.5061/dryad.9cnp5hr0w) · [Code](https://github.com/ericabusch/avatarRT_analysis)
  (verification: partly; see the research files)
- **NiNo: neuron graphs of weights**: Knyazev, B., Moudgil, A., Lajoie, G., Belilovsky, E., Lacoste-Julien, S. (2025). Accelerating Training with Neuron Interaction and Nowcasting Networks. ICLR 2025 (poster).
  [arXiv](https://arxiv.org/abs/2409.04434) · [Code](https://github.com/SamsungSAILMontreal/nino) · [Blog: Meta-Merging by Checkpoint Nowcasting (2026)](https://bknyaz.github.io/blog/2026/meta-merge/)
- **Phase transitions as level crossings**: Xu, Y. (2026). Spectral Edge Dynamics: an analytical-empirical study of phase transitions in neural network training. arXiv:2603.28964 (preprint, single author).
  [arXiv](https://arxiv.org/abs/2603.28964)
- **IPR of Hessian modes in training**: Marjankowska, M., Modugno, V., Barucca, P. (2026). Characterizing optimizer-dependent training dynamics through Hessian eigenvector displacement and localization. arXiv:2606.30226.
  [arXiv](https://arxiv.org/abs/2606.30226)
- **Varimax rotation**: H. F. Kaiser (1958). The varimax criterion for analytic rotation in factor analysis. Psychometrika 23(3).
  Used in: `nds.spectra.varimax_rotation`, `nds.spectra.aligned_localization`
  (verification: no; see the research files)
- **Varimax with guarantees**: K. Rohe, M. Zeng (2023). Vintage factor analysis with varimax performs statistical inference. Journal of the Royal Statistical Society Series B 85(4), 1068.
  [Journal](https://academic.oup.com/jrsssb/article/85/4/1068/7109146) · [arXiv](https://arxiv.org/abs/2004.05387)
  Used in: `nds.spectra.varimax_rotation`
- **Varimax asymptotics**: J. Cape. On varimax asymptotics in network models and spectral methods for dimensionality reduction. arXiv:2403.05461.
  [arXiv](https://arxiv.org/abs/2403.05461)
  (verification: partly; see the research files)
- **Spectral clustering by CPQR**: A. Damle, V. Minden, L. Ying (2019). Simple, direct and efficient multi-way spectral clustering. Information and Inference. arXiv:1609.08251.
  [arXiv](https://arxiv.org/abs/1609.08251)
  (verification: partly; see the research files)
- **Maximally localised Wannier functions**: N. Marzari, D. Vanderbilt (1997). Maximally localized generalized Wannier functions for composite energy bands. Physical Review B 56, 12847.
  [arXiv](https://arxiv.org/abs/cond-mat/9707145)
  Used in: `nds.timeline.aligned_module_vectors`, `nds.timeline.module_lineage`
- **Compressed modes**: V. Ozoliņš, R. Lai, R. Caflisch, S. Osher (2013). Compressed modes for variational problems in mathematics and physics. UCLA CAM report 13-45 (journal version in PNAS).
  [Report](https://ww3.math.ucla.edu/camreport/cam13-45.pdf)
  Used in: `nds.spectra.support_ratio`
  (verification: partly; see the research files)
- **Hoyer sparseness**: P. O. Hoyer (2004). Non-negative matrix factorization with sparseness constraints. Journal of Machine Learning Research 5.
  [JMLR](https://jmlr.org/papers/v5/hoyer04a.html)
  Used in: `nds.spectra.support_ratio`
- **Ground-state fidelity**: P. Zanardi, N. Paunković (2006). Ground state overlap and quantum phase transitions. Physical Review E 74, 031123.
  [arXiv](https://arxiv.org/abs/quant-ph/0512249)
  Used in: `nds.timeline.fidelity_susceptibility`, `nds.timeline.subspace_change_signal_to_noise`
- **Fidelity susceptibility review**: S.-J. Gu (2010). Fidelity approach to quantum phase transitions. International Journal of Modern Physics B 24. arXiv:0811.3127.
  [arXiv](https://arxiv.org/abs/0811.3127)
  Used in: `nds.timeline.fidelity_susceptibility`
  (verification: partly; see the research files)
- **ClassiFIM**: V. Kasatkin, E. Mozgunov, N. Ezzell, D. Lidar. Detecting quantum and classical phase transitions via unsupervised machine learning of the Fisher information metric. arXiv:2408.03418.
  [arXiv](https://arxiv.org/abs/2408.03418)
  (verification: partly; see the research files)
- **Grokking finite-size scaling**: Y. Bi, C. Zhang, Q. Wang, V. D. Calhoun (2026). Grokking as a falsifiable finite-size transition. arXiv:2603.24746.
  [arXiv](https://arxiv.org/abs/2603.24746)
  (verification: partly; see the research files)
- **Floating-point accuracy**: N. J. Higham (2002). Accuracy and Stability of Numerical Algorithms, 2nd edition. SIAM.
  (verification: no; see the research files)
- **Fidelity susceptibility by QMC**: L. Wang, Y.-H. Liu, J. Imriška, P. N. Ma, M. Troyer (2015). Fidelity susceptibility made simple: a unified quantum Monte Carlo approach. Physical Review X 5, 031007.
  [arXiv](https://arxiv.org/abs/1502.06969) · [DOI](https://doi.org/10.1103/PhysRevX.5.031007)
  (verification: partly; see the research files)
- **Common random numbers**: H. Kahn, A. W. Marshall (1953). Methods of reducing sample size in Monte Carlo computations. Journal of the Operations Research Society of America 1(5). Overview: Z. Botev, A. Ridder (2017). Variance reduction. Wiley StatsRef.
  [Wikipedia](https://en.wikipedia.org/wiki/Variance_reduction)
  Used in: `nds.analysis.paired_subspace_change`
  (verification: partly; see the research files)
- **The bootstrap**: B. Efron, R. J. Tibshirani (1993). An Introduction to the Bootstrap. Chapman & Hall.
  Used in: `nds.analysis.paired_subspace_change`, `nds.moments.BlockedNeuronMoments.bootstrap_replicate`
  (verification: no; see the research files)
- **Cross-validated distances (crossnobis)**: J. Diedrichsen, E. Berlot, M. Mur, H. H. Schütt, M. Shahbazi, N. Kriegeskorte (2021). Comparing representational geometries using whitened unbiased-distance-matrix similarity. Neurons, Behavior, Data analysis, and Theory. arXiv:2007.02789. (The estimator: Walther et al. 2016, NeuroImage.)
  [arXiv](https://arxiv.org/abs/2007.02789)
  Used in: `nds.spectra.crossvalidated_projector_distance`, `nds.analysis.crossvalidated_subspace_change`
  (verification: partly; see the research files)
- **Change points in network sequences**: D. Wang, Y. Yu, A. Rinaldo (2021). Optimal change point detection and localization in sparse dynamic networks. Annals of Statistics 49(1). arXiv:1809.09602.
  [arXiv](https://arxiv.org/abs/1809.09602)
  Used in: `nds.analysis.crossvalidated_subspace_change`
  (verification: partly; see the research files)
- **Bias of estimated projectors**: V. Koltchinskii, K. Lounici (2017). Normal approximation and concentration of spectral projectors of sample covariance. Annals of Statistics 45(1). arXiv:1504.07333.
  [arXiv](https://arxiv.org/abs/1504.07333)
  (verification: partly; see the research files)
- **CKA reliability (Wolf, Lajoie)**: M. Davari, S. Horoi, A. Natik, G. Lajoie, G. Wolf, E. Belilovsky (2023). Reliability of CKA as a similarity measure in deep learning. ICLR 2023. arXiv:2210.16156.
  [arXiv](https://arxiv.org/abs/2210.16156)
  (verification: partly; see the research files)
- **Permutation p-values**: B. Phipson, G. K. Smyth (2010). Permutation p-values should never be zero: calculating exact p-values when permutations are randomly drawn. Statistical Applications in Genetics and Molecular Biology 9(1), article 39. arXiv:1603.05766.
  [arXiv](https://arxiv.org/abs/1603.05766)
  Used in: `nds.worlds.world_rank_p_values`
- **Module preservation**: P. Langfelder, R. Luo, M. C. Oldham, S. Horvath (2011). Is my network module preserved and reproducible? PLoS Computational Biology 7(1): e1001057.
  [PLoS](https://doi.org/10.1371/journal.pcbi.1001057)
  (verification: partly; see the research files)

## VI. Round 12: sources behind the D67 tasks (added 7 Nov 2026, before any code)

Rule restated (AB, 7 Nov): every technique we use is referenced to where it came from, in this file
and in the credit header of the module that implements it, whether or not the source is in the
MAT 6495 review.

- **Degenerate and second-order perturbation theory** (P4.5, direct vs mediated coupling): Griffiths, D. J., Schroeter, D. F. (2018). Introduction to Quantum Mechanics, 3rd ed., ch. 7 (time-independent perturbation theory; degenerate case and the second-order energy). Sakurai, J. J., Napolitano, J. (2020). Modern Quantum Mechanics, 3rd ed., §5.1-5.2. The two-level model and Landau-Zener are credited in section I above (von Neumann & Wigner 1929; Zener 1932; Shevchenko, Ashhab & Nori 2010).
  The graph case that AB knows from optical rings: McGarvey-Lechable, K., Bianucci, P. (2018). Bloch-Floquet waves in optical ring resonators. [arXiv](https://arxiv.org/abs/1803.11312) (supplementary: mode-splitting derivation). McGarvey, K., Bianucci, P. (2022). Dielectric perturbations in optical ring resonators. Advanced Photonics Nexus 1, 016004.
  To be used in: `nds.flow` (new function next to `fit_avoided_crossing`), PLAN P4.5.
- **Eigenvalue multiplicities from symmetry** (P6.5a, degeneracy census): Serre, J.-P. (1977). Linear Representations of Finite Groups. Springer GTM 42, ch. 2 (characters; the regular representation contains each irrep with multiplicity equal to its dimension). Diaconis, P. (1988). Group Representations in Probability and Statistics, IMS Lecture Notes 11, ch. 3 (spectra of Cayley graphs when the generating set is a union of conjugacy classes). Chughtai, B., Chan, L., Nanda, N. (2023). A toy model of universality: reverse engineering how networks learn group operations. ICML 2023. [arXiv](https://arxiv.org/abs/2302.03025) (networks on group tasks learn irrep-based circuits).
  To be used in: aim 2 pilot (PLAN P6.5a), `nds.flow.group_near_degenerate_modes` for the counting.
- **Rotation-invariant quantities as the first channel, the neuron basis as the second** (P4.6 and the P6.1 paragraph): the regression itself is ordinary least squares; the framing comes from the identifiability problem in resonator spectroscopy: Couillard, A., Bianucci, P. (2022). Absolute radius, index and dispersion of a fibre from resonance wavelengths. [arXiv](https://arxiv.org/abs/2205.02407). Deep, Eadie & Bianucci (2026), cavity optomechanics in an infinite cylinder, [arXiv](https://arxiv.org/abs/2605.29154) §I. Baselines: Li et al. 2025 (RankMe, alpha-ReQ), credited in section III.
- **Persistence of localisation across kernel bandwidth** (P4.7): Hata, S., Nakao, H. (2017). Localization of Laplacian eigenvectors on random networks. Scientific Reports 7: 1121 (credited in section I; this task is a test built around their mechanism). The persistence idea (a robust mode survives smooth deformation): Asboth, J. K., Oroszlany, L., Palyi, A. (2016). A Short Course on Topological Insulators, Springer Lecture Notes in Physics 919, ch. 1. [arXiv](https://arxiv.org/abs/1509.02295). Used as motivation only; no topological claim is made.
- **Diffusion time as a low-pass filter** (P6.1 paragraph): Coifman, R. R., Lafon, S. (2006). Diffusion maps. Applied and Computational Harmonic Analysis 21(1): 5-30, §2-3 (credited in section II). The absorption analogy: Khattak, H. K., Bianucci, P., Slepkov, A. D. (2019). Linking plasma formation in grapes to microwave resonances of aqueous dimers. PNAS 116(10): 4000-4005 (AB ran the simulations for this paper).
- **Structured vs diffuse spectral change** (P6.5b, one sentence): participation ratio as a measure of how many modes carry a change: Bell, R. J., Dean, P. (1970). Atomic vibrations in vitreous silica. Discussions of the Faraday Society 50: 55-61 (the original participation ratio). Eadie, Deep & Bianucci (2025). Azimuthal Kerr combs with predictable mode spacing in SNAP microresonators. Optics Express 33(16) (the single- vs whole-circumference perturbation contrast that suggested it).

## Code repositories read for this project

| repository | licence | what we take |
|---|---|---|
| [Aaquib111/edge-attribution-patching](https://github.com/Aaquib111/edge-attribution-patching) | none | No license: read only. Use the maintained hannamw/EAP-IG instead. |
| [ArthurConmy/Automatic-Circuit-Discovery](https://github.com/ArthurConmy/Automatic-Circuit-Discovery) | MIT | MIT text. Cite as the reference method; too slow to run at 154 checkpoints. |
| [bayazitdeniz/crosscoding-through-time](https://github.com/bayazitdeniz/crosscoding-through-time) | MIT | MIT; read for feature-level tracking over training. Pins torch 2.5.1, transformer_lens 2.8.1, nnsight 0.4.6, sae-lens 5.2.0; Pythia-1B only. |
| [CalculatedContent/WeightWatcher](https://github.com/CalculatedContent/WeightWatcher) | Apache-2.0 | Apache-2.0: may adapt with credit. Read WW_powerlaw.py for the fit; run the tool itself on a few checkpoints to cross-check nds.baselines.htsr_alpha. |
| [chengemily1/id-llm-abstraction](https://github.com/chengemily1/id-llm-abstraction) | MIT | MIT. Checkpoint loading must be added before use on Pythia steps. |
| [ChenLiu-1996/DiffusionSpectralEntropy](https://github.com/ChenLiu-1996/DiffusionSpectralEntropy) | Yale non-commercial | Yale non-commercial: do not copy code into our repo; pip-install for checks, or reimplement the (short) formulas. Our formula equals theirs to 9e-16; keep dse_dsmi as a test oracle only. |
| [ChenLiu-1996/LM-Dispersion](https://github.com/ChenLiu-1996/LM-Dispersion) | Yale non-commercial | Yale non-commercial: read and quote with credit, never copy. Our nds.token_sample.mean_token_cosine is re-implemented from the definition. |
| [curt-tigges/circuits-over-time](https://github.com/curt-tigges/circuits-over-time) | MIT | MIT. Not kept locally (787 MB); re-clone with --filter=blob:none --sparse if needed. Needs EAP-IG at commit 10035b88 in edge_attribution_patching/. |
| [danspielman/sagt_code](https://github.com/danspielman/sagt_code) | MIT | MIT: copying with the notice is allowed, but it is Julia; use it to reproduce the book's pictures while studying for MAT 6495, not as project code. |
| [DavyMorgan/llm-graph-probing](https://github.com/DavyMorgan/llm-graph-probing) | Apache-2.0 | Apache-2.0: copying allowed with the notice and a statement of changes. We take the idea only; our graph differs on four points. |
| [decoderesearch/circuit-tracer](https://github.com/decoderesearch/circuit-tracer) | MIT | MIT (copy with notice). Reading material and a possible viewer for exported supernodes; needs nnsight 0.8 and transformers 4.56 to 4.57.3, so a separate environment. |
| [decoderesearch/SAELens](https://github.com/decoderesearch/SAELens) | MIT | MIT. Alternative download of the Marks dictionaries; transpose W_enc before using our bridge. |
| [deepcharles/ruptures](https://github.com/deepcharles/ruptures) | BSD-2-Clause | BSD-2: copying with the notice is allowed, but we just import it (pip install ruptures). Read pelt.py and costl2.py to understand the penalty scale. |
| [diegodoimo/geometry_representations](https://github.com/diegodoimo/geometry_representations) | none | No LICENSE file; background reading only. |
| [ejmichaud/quantization-model](https://github.com/ejmichaud/quantization-model) | MIT | MIT. Clusters contexts (behaviours), not neurons; whether the exact Marks 70m-deduped run is public is unverified. |
| [EleutherAI/pythia](https://github.com/EleutherAI/pythia) | Apache-2.0 | Apache-2.0. Read the README and the 70m config; our nds.checkpoints matches its step list and branch names. |
| [EleutherAI/sparsify](https://github.com/EleutherAI/sparsify) | MIT | MIT. The route to dictionaries on layers.*.mlp.act (our 2,048 neurons) if ever needed; works with plain transformers. |
| [ericabusch/tphate_analysis_capsule](https://github.com/ericabusch/tphate_analysis_capsule) | GPL-3.0 | GPL-3.0: do not copy; read only as an example of a full PHATE-family analysis pipeline. |
| [hannamw/EAP-IG](https://github.com/hannamw/EAP-IG) | MIT | MIT. Its 'neuron' level is d_model output dimensions, so it does not replace our mlp.act attribution; needs transformer_lens 2.x in its own environment. |
| [hannamw/eap-ig-faithfulness](https://github.com/hannamw/eap-ig-faithfulness) | none | No license: read only. |
| [hematteo/learning-to-read-out](https://github.com/hematteo/learning-to-read-out) | MIT | MIT: may adapt with credit. We re-implemented the null in nds.timeline.circular_shift_p_value; read find_reorganization_steps.py lines 266-310. |
| [huggingface/transformers](https://github.com/huggingface/transformers) | not checked (LICENSE not in the sparse clone) | Our dependency (transformers >=4.40); hooks on mlp.act give the 2,048 neurons. |
| [jacobdunefsky/transcoder_circuits](https://github.com/jacobdunefsky/transcoder_circuits) | none | No license (sae_training's upstream license unverified): read and cite only. get_transcoder_ixg is the template for our W_out bridge. |
| [JianchengXie/MMPHATE_REPRO](https://github.com/JianchengXie/MMPHATE_REPRO) | none | No LICENSE file (pyproject.toml says MIT): do not copy without asking; link and quote only. Useful to see that the Sec. 5 entropy analysis is not in the released code. |
| [jinhl9/llm-compositionality-lifetime](https://github.com/jinhl9/llm-compositionality-lifetime) | MIT | MIT. Fix the missing revision= before reusing; the CSV is directly usable for an ID-versus-step overlay. |
| [kaistAI/Knowledge-Entropy](https://github.com/kaistAI/Knowledge-Entropy) | Apache-2.0 | Apache-2.0 (LICENSE file checked): may adapt with credit; the formula is re-implemented in nds.baselines.knowledge_entropy. |
| [KrishnaswamyLab/CATCH](https://github.com/KrishnaswamyLab/CATCH) | Yale non-commercial | Yale non-commercial: do not copy code into our repo; pip-install for checks, or reimplement the (short) formulas. Note the reused variable t_max (line 352) if reading it closely. |
| [KrishnaswamyLab/diffusion_condensation](https://github.com/KrishnaswamyLab/diffusion_condensation) | GPL-3.0 | GPL-3.0: do not copy. Reimplement the ~30-line loop for part 2 with a Gaussian step kernel and deterministic epsilon; the package runs on the PHATE potential, not on raw neuron profiles. |
| [KrishnaswamyLab/DiffusionEMD](https://github.com/KrishnaswamyLab/DiffusionEMD) | Yale non-commercial | Yale non-commercial: do not copy code into our repo; pip-install for checks, or reimplement the (short) formulas. If used, pass a scipy.sparse adjacency and note its default anisotropy = 1. |
| [KrishnaswamyLab/graphtools](https://github.com/KrishnaswamyLab/graphtools) | GPL-3.0 | GPL-3.0: read and reimplement from the formula (as nds.graphs does with kernel_type='alpha_decay'); do not copy code. |
| [KrishnaswamyLab/HeatGeo](https://github.com/KrishnaswamyLab/HeatGeo) | Yale non-commercial | Yale non-commercial: do not copy code into our repo; pip-install for checks, or reimplement the (short) formulas. Only the Chebyshev filter would ever matter, and only for graphs too large for dense eigh. |
| [KrishnaswamyLab/Multiscale_PHATE](https://github.com/KrishnaswamyLab/Multiscale_PHATE) | Yale non-commercial | Yale non-commercial: do not copy code into our repo; pip-install for checks, or reimplement the (short) formulas. The level-picking rule (get_levels) is the part worth reading. |
| [KrishnaswamyLab/NN-LossLandscape-PHATE](https://github.com/KrishnaswamyLab/NN-LossLandscape-PHATE) | Yale non-commercial | Yale non-commercial: do not copy code into our repo; pip-install for checks, or reimplement the (short) formulas. README credits MIT-licensed pluskid/fitting-random-labels and tomgoldstein/loss-landscape for parts of the code. |
| [KrishnaswamyLab/PHATE](https://github.com/KrishnaswamyLab/PHATE) | Yale non-commercial | Yale non-commercial (LICENSE.md; vne.py's header still says GPLv2): read only. See docs/method-notes.md §4 for how its entropy relates to our DSE. Covered in more depth in Part II. |
| [KrishnaswamyLab/TPHATE](https://github.com/KrishnaswamyLab/TPHATE) | Yale non-commercial | Yale non-commercial: do not copy code into our repo; pip-install for checks, or reimplement the (short) formulas. Do not import next to graphtools. |
| [kumarkrishna/fastssl](https://github.com/kumarkrishna/fastssl) | none | No LICENSE file, so all rights reserved: quote for commentary only, do not copy files. |
| [locuslab/massive-activations](https://github.com/locuslab/massive-activations) | MIT | MIT. Only experiment 3 runs for Pythia; the useful part is the per-token rule from the paper, which we should add next to our per-neuron rule. |
| [mattf1n/lm-intervention](https://github.com/mattf1n/lm-intervention) | MIT | MIT. Credit it alongside Marks et al. whenever we use the SVA data; requires transformers==2.7, so read, do not run. |
| [melodylizx/llm-geometry-project](https://github.com/melodylizx/llm-geometry-project) | none | Nothing to reuse; read the page for the paper's figures. |
| [melodylizx/Tracing-the-representation-geometry-of-language-models](https://github.com/melodylizx/Tracing-the-representation-geometry-of-language-models) | none | Quote short snippets with credit and re-implement (done: nds.baselines). The data/*_loader.py modules are missing, so the scripts do not run as cloned. |
| [MouseLand/stringer-pachitariu-et-al-2018b](https://github.com/MouseLand/stringer-pachitariu-et-al-2018b) | GPL-3.0 | Cite for the method and keep our own implementation; pasting the function would bring GPL-3.0 into our repo. |
| [ndif-team/nnsight](https://github.com/ndif-team/nnsight) | MIT | MIT. Only needed for part 2 tools; the revision path was read, not run. |
| [OFSkean/information_flow](https://github.com/OFSkean/information_flow) | none | No LICENSE file: re-implement (done: nds.baselines.prompt_entropy), do not copy. |
| [rkillick/changepoint](https://github.com/rkillick/changepoint) | GPL | GPL (DESCRIPTION; no LICENSE file): read only, do not copy. Useful to cross-check ruptures on one series and to see the standard penalties. |
| [SamsungSAILMontreal/nino](https://github.com/SamsungSAILMontreal/nino) | MIT | MIT (LICENSE file checked): may adapt with credit. Read graph/graph.py to see how a weight graph over neurons is assembled before building a weight-side pythia-70m graph. |
| [saprmarks/dictionary_learning](https://github.com/saprmarks/dictionary_learning) | MIT | MIT. Load dictionaries with AutoEncoder.from_pretrained so magnitudes match saved circuits; encoder.weight (32768, 512) feeds nds.naming.feature_loadings_on_neurons. |
| [saprmarks/feature-circuits](https://github.com/saprmarks/feature-circuits) | MIT | MIT: may copy load_examples and the SVA data with the notice (credit Finlayson et al. too). Needs nnsight 0.3.7 and local fixes (ablation.py import, evaluate_circuit.sh flags, mkdir circuits/) if run. |
| [scottgigante/m-phate](https://github.com/scottgigante/m-phate) | GPL-3.0 | GPL-3.0: do not copy. Reimplement the interslice formula (about 10 lines) if we build a joint checkpoint x neuron graph; if ever run on wide probe sets, set n_pca=None. |
| [sissa-data-science/DADApy](https://github.com/sissa-data-science/DADApy) | Apache-2.0 | Apache-2.0, a library to call, not copy. The keyword names changed (mu_fraction, data_fraction), so older call sites break. |
| [Superposition09m/SAE-Track](https://github.com/Superposition09m/SAE-Track) | none | No license: read and cite only, do not copy. Bundles old copies of sae_lens and transformer_lens. |
| [THUNLP/modularity-analysis](https://github.com/THUNLP/modularity-analysis) | MIT | MIT. Copy the random-balanced-partition null idea for module-stability claims; data comes from Aliyun tarballs. |
| [timaeus-research/devinterp](https://github.com/timaeus-research/devinterp) | MIT | MIT. A possible LLC baseline per checkpoint; needs a GPU and gradient sampling. |
| [TransformerLensOrg/TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) | MIT | MIT. Not needed for our collection (D3); pin transformer_lens==2.18.0 in a separate environment for code that needs HookedTransformer. |
| [TransluceAI/circuits](https://github.com/TransluceAI/circuits) | none | No license at the root (only the frontend assets are MIT, Anthropic): do not copy; re-implement from arXiv 2601.22594 / 2604.07615 and cite. |
| [uk-cliplab/representation-itl](https://github.com/uk-cliplab/representation-itl) | none | No LICENSE file (setup.cfg classifier says MIT). Avoid its Renyi order 2 path: on a trace-normalised matrix it is too large by 2 log N. |
| [wesg52/universal-neurons](https://github.com/wesg52/universal-neurons) | MIT | MIT, so copying is allowed with the notice, but its fp16 raw-sum numerics are worse than nds.moments; copy only the token filter idea. |

## Data and models

- Pythia model suite and checkpoints: EleutherAI, https://huggingface.co/EleutherAI (Biderman et al. 2023).
- PolyPythias seeds: https://huggingface.co/collections/EleutherAI/polypythias (van der Wal et al. 2025).
- Probe text: the Pile validation split (Gao et al. 2020) via the mirror https://huggingface.co/datasets/monology/pile-uncopyrighted .
- Subject-verb agreement pairs (part 2): Marks et al. (2025), https://github.com/saprmarks/feature-circuits (MIT).

## Software

NumPy, SciPy, PyTorch, Hugging Face transformers and huggingface_hub, ruptures (Truong et al. 2020), matplotlib, PyYAML, zstandard, pytest; KaTeX for the encyclopedia's maths.
