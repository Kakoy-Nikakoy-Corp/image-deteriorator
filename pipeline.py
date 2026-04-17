import albumentations as A

pipeline = A.Compose(
    [
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=10, p=0.4
        ),
        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.3),
        A.RandomGamma(gamma_limit=(85, 115), p=0.3),
        A.ImageCompression(quality_range=(60, 90), p=0.3),
        A.AtmosphericFog(density_range=(0.5, 1.5), depth_mode="linear", p=0.3),
        A.ChromaticAberration(
            primary_distortion_limit=0.002,
            secondary_distortion_limit=0.002,
            mode="green_purple",
            interpolation=1,
            p=0.2,
        ),
        A.OpticalDistortion(distort_limit=0.05, p=0.25),
        A.OneOf(
            [
                A.Downscale(scale_range=(0.75, 0.9)),
                A.Defocus(radius=(1, 3), alias_blur=(0.1, 0.3)),
            ],
            p=0.3,
        ),
        A.OneOf(
            [
                A.ISONoise(color_shift=(0.01, 0.02), intensity=(0.1, 0.25)),
                A.GaussNoise(std_range=(0.05, 0.1)),
                A.ShotNoise(scale_range=(0.01, 0.04)),
            ],
            p=0.3,
        ),
        A.ToGray(p=0.25),
    ]
)
