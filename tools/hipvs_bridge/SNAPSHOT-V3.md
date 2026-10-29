# GPU snapshot V3

V3 removes the outer duplicate FP32 vector matrix. IDs and the complete native CAGRA payload remain persisted; that native payload includes the FP32 dataset. On reload, `seekdb_cuvs_copy_dataset` validates the native tensor and restores the CPU matrix before publishing the index. It preserves row pitch, byte offset and original FP32 bits.

The new reader accepts both `OBCUVS02` and `OBCUVS03`. Old binaries cannot read V3, so rollback to an old binary requires an old accepted database copy or a fresh rebuild. No frozen SDK source, external SQL, input precision or scoring file is changed.

The ELF and bridge must be built together: the new ELF requires the bridge's `seekdb_cuvs_copy_dataset` symbol. Old experiment bridge libraries do not implement that ABI.

Source review and an exported SDK symbol check are complete. Native compilation, real-GPU copy verification and SQL persistence/performance validation must be completed before this change is accepted. Development measurements are not official scores.
