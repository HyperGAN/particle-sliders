"""Explicit runtime boundary for the model-independent core distribution."""
from pathlib import Path

from setuptools.command.build_py import build_py


CORE_MODULES = frozenset({
    '__init__', '_restore_training', '_training',
    'adapter_activation', 'adapter_attachment', 'adapter_loading', 'bispf_game',
    'distillation', 'endpoint_game', 'field_critic', 'formulation',
    'formulation_game', 'grad_regularizers', 'inference', 'legacy_compat',
    'lifecycle', 'native_lora', 'native_lora_compat', 'optimizer_policy', 'peft_scaling', 'pg09',
    'pg093', 'pg094', 'pg095', 'pg09_audio', 'pg09_backends', 'pg09_hndl',
    'pg09_readout_critic', 'recipe', 'reference', 'response',
    'restoration', 'routed_projection', 'session_cli', 'teachers', 'weight_policy',
})


class CoreBuildPy(build_py):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        if package == 'particle_sliders':
            return [item for item in modules if item[1] in CORE_MODULES]
        return modules

    def run(self):
        super().run()
        # A previous embedded-product build can leave files in build/lib.
        # install_lib copies that whole directory, so trim stale outputs too.
        outputs = {Path(name).absolute() for name in self.get_outputs(include_bytecode=False)}
        root = Path(self.build_lib)
        if root.exists():
            for path in root.rglob('*'):
                if (path.is_file() or path.is_symlink()) and path.absolute() not in outputs:
                    path.unlink()
