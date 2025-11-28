from dmmd import dmmd_blockwise

def compute_palate(train_representations, test_representations, gen_representations):

    dmmd_train, _ = dmmd_blockwise(train_representations, gen_representations)
    dmmd_test, denominator_scale = dmmd_blockwise(
        test_representations, gen_representations
    )

    palate = dmmd_test / (dmmd_test + dmmd_train)
    m_palate = dmmd_test / (2 * denominator_scale) + (1 / 2) * palate
    return m_palate, palate
