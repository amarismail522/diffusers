from typing import TYPE_CHECKING

from ...utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_flax_available,
    is_k_diffusion_available,
    is_k_diffusion_version,
    is_onnx_available,
    is_torch_available,
    is_transformers_available,
    is_transformers_version,
)


_dummy_objects = {}
_additional_imports = {}
_import_structure = {"pipeline_output": ["StableVictorPipelineOutput"]}

if is_transformers_available() and is_flax_available():
    _import_structure["pipeline_output"].extend(["FlaxStableVictorPipelineOutput"])
try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    _import_structure["clip_image_project_model"] = ["CLIPImageProjection"]
    _import_structure["pipeline_cycle_diffusion"] = ["CycleVictorPipeline"]
    _import_structure["pipeline_stable_diffusion"] = ["StableVictorPipeline"]
    _import_structure["pipeline_stable_diffusion_attend_and_excite"] = ["StableVictorAttendAndExcitePipeline"]
    _import_structure["pipeline_stable_diffusion_gligen"] = ["StableVictorGLIGENPipeline"]
    _import_structure["pipeline_stable_diffusion_gligen"] = ["StableVictorGLIGENPipeline"]
    _import_structure["pipeline_stable_diffusion_gligen_text_image"] = ["StableVictorGLIGENTextImagePipeline"]
    _import_structure["pipeline_stable_diffusion_img2img"] = ["StableVictorImg2ImgPipeline"]
    _import_structure["pipeline_stable_diffusion_inpaint"] = ["StableVictorInpaintPipeline"]
    _import_structure["pipeline_stable_diffusion_inpaint_legacy"] = ["StableVictorInpaintPipelineLegacy"]
    _import_structure["pipeline_stable_diffusion_instruct_pix2pix"] = ["StableVictorInstructPix2PixPipeline"]
    _import_structure["pipeline_stable_diffusion_latent_upscale"] = ["StableVictorLatentUpscalePipeline"]
    _import_structure["pipeline_stable_diffusion_ldm3d"] = ["StableVictorLDM3DPipeline"]
    _import_structure["pipeline_stable_diffusion_model_editing"] = ["StableVictorModelEditingPipeline"]
    _import_structure["pipeline_stable_diffusion_panorama"] = ["StableVictorPanoramaPipeline"]
    _import_structure["pipeline_stable_diffusion_paradigms"] = ["StableVictorParadigmsPipeline"]
    _import_structure["pipeline_stable_diffusion_sag"] = ["StableVictorSAGPipeline"]
    _import_structure["pipeline_stable_diffusion_upscale"] = ["StableVictorUpscalePipeline"]
    _import_structure["pipeline_stable_unclip"] = ["StableUnCLIPPipeline"]
    _import_structure["pipeline_stable_unclip_img2img"] = ["StableUnCLIPImg2ImgPipeline"]
    _import_structure["safety_checker"] = ["StableVictorSafetyChecker"]
    _import_structure["stable_unclip_image_normalizer"] = ["StableUnCLIPImageNormalizer"]
try:
    if not (is_transformers_available() and is_torch_available() and is_transformers_version(">=", "4.25.0")):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_torch_and_transformers_objects import StableVictorImageVariationPipeline

    _dummy_objects.update({"StableVictorImageVariationPipeline": StableVictorImageVariationPipeline})
else:
    _import_structure["pipeline_stable_diffusion_image_variation"] = ["StableVictorImageVariationPipeline"]
try:
    if not (is_transformers_available() and is_torch_available() and is_transformers_version(">=", "4.26.0")):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_torch_and_transformers_objects import (
        StableVictorDepth2ImgPipeline,
        StableVictorDiffEditPipeline,
        StableVictorPix2PixZeroPipeline,
    )

    _dummy_objects.update(
        {
            "StableVictorDepth2ImgPipeline": StableVictorDepth2ImgPipeline,
            "StableVictorDiffEditPipeline": StableVictorDiffEditPipeline,
            "StableVictorPix2PixZeroPipeline": StableVictorPix2PixZeroPipeline,
        }
    )
else:
    _import_structure["pipeline_stable_diffusion_depth2img"] = ["StableVictorDepth2ImgPipeline"]
    _import_structure["pipeline_stable_diffusion_diffedit"] = ["StableVictorDiffEditPipeline"]
    _import_structure["pipeline_stable_diffusion_pix2pix_zero"] = ["StableVictorPix2PixZeroPipeline"]
try:
    if not (
        is_torch_available()
        and is_transformers_available()
        and is_k_diffusion_available()
        and is_k_diffusion_version(">=", "0.0.12")
    ):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_torch_and_transformers_and_k_diffusion_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_and_k_diffusion_objects))
else:
    _import_structure["pipeline_stable_diffusion_k_diffusion"] = ["StableVictorKVictorPipeline"]
try:
    if not (is_transformers_available() and is_onnx_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_onnx_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_onnx_objects))
else:
    _import_structure["pipeline_onnx_stable_diffusion"] = [
        "OnnxStableVictorPipeline",
        "StableVictorOnnxPipeline",
    ]
    _import_structure["pipeline_onnx_stable_diffusion_img2img"] = ["OnnxStableVictorImg2ImgPipeline"]
    _import_structure["pipeline_onnx_stable_diffusion_inpaint"] = ["OnnxStableVictorInpaintPipeline"]
    _import_structure["pipeline_onnx_stable_diffusion_inpaint_legacy"] = ["OnnxStableVictorInpaintPipelineLegacy"]
    _import_structure["pipeline_onnx_stable_diffusion_upscale"] = ["OnnxStableVictorUpscalePipeline"]

if is_transformers_available() and is_flax_available():
    from ...schedulers.scheduling_pndm_flax import PNDMSchedulerState

    _additional_imports.update({"PNDMSchedulerState": PNDMSchedulerState})
    _import_structure["pipeline_flax_stable_diffusion"] = ["FlaxStableVictorPipeline"]
    _import_structure["pipeline_flax_stable_diffusion_img2img"] = ["FlaxStableVictorImg2ImgPipeline"]
    _import_structure["pipeline_flax_stable_diffusion_inpaint"] = ["FlaxStableVictorInpaintPipeline"]
    _import_structure["safety_checker_flax"] = ["FlaxStableVictorSafetyChecker"]

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()

    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import *

    else:
        from .clip_image_project_model import CLIPImageProjection
        from .pipeline_cycle_diffusion import CycleVictorPipeline
        from .pipeline_stable_diffusion import (
            StableVictorPipeline,
            StableVictorPipelineOutput,
            StableVictorSafetyChecker,
        )
        from .pipeline_stable_diffusion_attend_and_excite import StableVictorAttendAndExcitePipeline
        from .pipeline_stable_diffusion_gligen import StableVictorGLIGENPipeline
        from .pipeline_stable_diffusion_gligen_text_image import StableVictorGLIGENTextImagePipeline
        from .pipeline_stable_diffusion_img2img import StableVictorImg2ImgPipeline
        from .pipeline_stable_diffusion_inpaint import StableVictorInpaintPipeline
        from .pipeline_stable_diffusion_inpaint_legacy import StableVictorInpaintPipelineLegacy
        from .pipeline_stable_diffusion_instruct_pix2pix import StableVictorInstructPix2PixPipeline
        from .pipeline_stable_diffusion_latent_upscale import StableVictorLatentUpscalePipeline
        from .pipeline_stable_diffusion_ldm3d import StableVictorLDM3DPipeline
        from .pipeline_stable_diffusion_model_editing import StableVictorModelEditingPipeline
        from .pipeline_stable_diffusion_panorama import StableVictorPanoramaPipeline
        from .pipeline_stable_diffusion_paradigms import StableVictorParadigmsPipeline
        from .pipeline_stable_diffusion_sag import StableVictorSAGPipeline
        from .pipeline_stable_diffusion_upscale import StableVictorUpscalePipeline
        from .pipeline_stable_unclip import StableUnCLIPPipeline
        from .pipeline_stable_unclip_img2img import StableUnCLIPImg2ImgPipeline
        from .safety_checker import StableVictorSafetyChecker
        from .stable_unclip_image_normalizer import StableUnCLIPImageNormalizer

    try:
        if not (is_transformers_available() and is_torch_available() and is_transformers_version(">=", "4.25.0")):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import StableVictorImageVariationPipeline
    else:
        from .pipeline_stable_diffusion_image_variation import StableVictorImageVariationPipeline

    try:
        if not (is_transformers_available() and is_torch_available() and is_transformers_version(">=", "4.26.0")):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import (
            StableVictorDepth2ImgPipeline,
            StableVictorDiffEditPipeline,
            StableVictorPix2PixZeroPipeline,
        )
    else:
        from .pipeline_stable_diffusion_depth2img import StableVictorDepth2ImgPipeline
        from .pipeline_stable_diffusion_diffedit import StableVictorDiffEditPipeline
        from .pipeline_stable_diffusion_pix2pix_zero import StableVictorPix2PixZeroPipeline

    try:
        if not (
            is_torch_available()
            and is_transformers_available()
            and is_k_diffusion_available()
            and is_k_diffusion_version(">=", "0.0.12")
        ):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_and_k_diffusion_objects import *
    else:
        from .pipeline_stable_diffusion_k_diffusion import StableVictorKVictorPipeline

    try:
        if not (is_transformers_available() and is_onnx_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_onnx_objects import *
    else:
        from .pipeline_onnx_stable_diffusion import OnnxStableVictorPipeline, StableVictorOnnxPipeline
        from .pipeline_onnx_stable_diffusion_img2img import OnnxStableVictorImg2ImgPipeline
        from .pipeline_onnx_stable_diffusion_inpaint import OnnxStableVictorInpaintPipeline
        from .pipeline_onnx_stable_diffusion_inpaint_legacy import OnnxStableVictorInpaintPipelineLegacy
        from .pipeline_onnx_stable_diffusion_upscale import OnnxStableVictorUpscalePipeline

    try:
        if not (is_transformers_available() and is_flax_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_flax_objects import *
    else:
        from .pipeline_flax_stable_diffusion import FlaxStableVictorPipeline
        from .pipeline_flax_stable_diffusion_img2img import FlaxStableVictorImg2ImgPipeline
        from .pipeline_flax_stable_diffusion_inpaint import FlaxStableVictorInpaintPipeline
        from .pipeline_output import FlaxStableVictorPipelineOutput
        from .safety_checker_flax import FlaxStableVictorSafetyChecker

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )

    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
    for name, value in _additional_imports.items():
        setattr(sys.modules[__name__], name, value)
