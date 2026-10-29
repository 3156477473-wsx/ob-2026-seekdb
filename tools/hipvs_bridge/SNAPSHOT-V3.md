# GPU snapshot V3

V3 removes the outer duplicate FP32 vector matrix. IDs and the complete native CAGRA payload remain persisted; that native payload includes the FP32 dataset. On reload, `seekdb_cuvs_copy_dataset` validates the native tensor and restores the CPU matrix before publishing the index. It preserves row pitch, byte offset and original FP32 bits.

The new reader accepts both `OBCUVS02` and `OBCUVS03`. Old binaries cannot read V3, so rollback to an old binary requires an old accepted database copy or a fresh rebuild. No frozen SDK source, external SQL, input precision or scoring file is changed.

The ELF and bridge must be built together: the new ELF requires the bridge's `seekdb_cuvs_copy_dataset` symbol. Old experiment bridge libraries do not implement that ABI.

The default NN-descent intermediate graph degree is 160, with final degree 128, iteration cap 100 and query settings unchanged. The 60a native host with this exact bridge configuration passed full 250K/4096-query Gaussian tests at dimensions 128 and 256, Recall/query checks, sampled VRAM against a measured frozen V5 reference, and normal reopen checks. Native V3 copy tests also verified FP32 bits at dimensions 128, 256 and 129. The final integrated commit still requires its own build and public-SIFT verification. Development measurements are not official scores.
