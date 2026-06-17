# Creating and Training a Custom Architecture in Megatron & Megatron-Bridge

To train a custom deep learning architecture in Megatron-LM and export it to a Hugging Face (HF) checkpoint, you can leverage **NeMo Megatron-Bridge**. Megatron-Bridge serves as the interoperability, conversion, and verification layer between Hugging Face and **Megatron Core (MCore)**.

This guide provides a detailed, end-to-end walkthrough of the 5 phases required to build, map, train, and export your own custom architecture.

---

## Architecture Translation Workflow

```mermaid
graph TD
    subgraph HF ["Hugging Face Ecosystem"]
        HF_Model["Custom HF Model<br>(modeling_custom.py)"]
        HF_Config["config.json"]
        HF_CKPT["HF Checkpoint<br>(safetensors)"]
    end

    subgraph Bridge ["NeMo Megatron-Bridge (Interoperability Layer)"]
        Bridge_Def["Custom Bridge Class<br>(@register_bridge)"]
        Weight_Map["Weight Mapping Registry<br>(AutoMapping, QKVMapping...)"]
        Provider_Bridge["Provider Config Translation"]
    end

    subgraph MCore ["Megatron Core & Megatron-LM"]
        MC_Provider["Model Provider<br>(GPTModelProvider)"]
        MC_Spec["Module Spec Configuration<br>(ModuleSpec)"]
        MC_Model["Megatron Model<br>(GPTModel / Custom)"]
        MC_Train["Distributed Training Loop<br>(pretrain / SFT)"]
        MC_CKPT["Megatron Checkpoint<br>(Distributed Shards)"]
    end

    HF_Config -->|Config Mapping| Bridge_Def
    HF_Model -->|Source Registration| Bridge_Def
    Bridge_Def -->|Finalizes & Configures| MC_Provider
    MC_Provider -->|Instantiates| MC_Model
    MC_Spec -->|Composes Layers of| MC_Model
    
    HF_CKPT -->|1. Import (convert_checkpoints.py)| MC_CKPT
    MC_CKPT -->|2. Distributed Training| MC_Train
    MC_Train -->|Saves updated| MC_CKPT
    MC_CKPT -->|3. Export (convert_checkpoints.py)| HF_CKPT
```

---

## Phase 1: Defining the Custom Hugging Face Model

Your architecture must first exist as a standard PyTorch model class integrated into Hugging Face `transformers`.

1. **Create the HF Configuration Class** (`configuration_custom.py`):
   ```python
   from transformers import PretrainedConfig

   class CustomConfig(PretrainedConfig):
       model_type = "custom_transformer"

       def __init__(
           self,
           vocab_size=32000,
           hidden_size=2048,
           num_hidden_layers=24,
           num_attention_heads=16,
           num_key_value_heads=8,
           intermediate_size=5632,
           rms_norm_eps=1e-6,
           tie_word_embeddings=False,
           rope_theta=10000.0,
           my_custom_feature_param=42,  # Custom architecture field
           **kwargs,
       ):
           super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
           self.vocab_size = vocab_size
           self.hidden_size = hidden_size
           self.num_hidden_layers = num_hidden_layers
           self.num_attention_heads = num_attention_heads
           self.num_key_value_heads = num_key_value_heads
           self.intermediate_size = intermediate_size
           self.rms_norm_eps = rms_norm_eps
           self.rope_theta = rope_theta
           self.my_custom_feature_param = my_custom_feature_param
   ```

2. **Create the HF Modeling Class** (`modeling_custom.py`):
   ```python
   import torch
   from transformers import PreTrainedModel
   from .configuration_custom import CustomConfig

   class CustomModelForCausalLM(PreTrainedModel):
       config_class = CustomConfig
       base_model_prefix = "model"

       def __init__(self, config: CustomConfig):
           super().__init__(config)
           # Define embeddings, custom transformer layers, norms, and lm_head
           ...

       def forward(self, input_ids, labels=None, **kwargs):
           # Standard PyTorch forward pass returning CausalLMOutputWithPast
           ...
   ```

---

## Phase 2: Defining the Model in Megatron Core

Megatron Core (MCore) constructs models using a **spec-based composition** pattern (`ModuleSpec`). This decouples the structure of a model (e.g. sequence of blocks, attention steps) from the exact implementation details (e.g. standard local layers vs. optimized Transformer Engine layers).

You can configure your custom architecture in Megatron using two main approaches:

### Approach A: Composing Submodules via `ModuleSpec` (Recommended)
If your architecture is a variant of the standard transformer (e.g. custom attention, custom MLP block, or a hybrid layout), you can use Megatron-LM's built-in `GPTModel` but provide a custom `transformer_layer_spec`.

1. Define your custom attention or MLP layer as a subclass of `nn.Module` using Megatron Core parallel primitives:
   ```python
   from torch import nn
   from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear

   class CustomParallelMLP(nn.Module):
       def __init__(self, config, pg_collection=None):
           super().__init__()
           # ColumnParallelLinear splits weights along output dimension
           self.linear_fc1 = ColumnParallelLinear(
               config.hidden_size,
               config.ffn_hidden_size,
               config=config,
               init_method=config.init_method,
               bias=False,
               tp_group=pg_collection.tp
           )
           # RowParallelLinear splits weights along input dimension (reducing TP ranks)
           self.linear_fc2 = RowParallelLinear(
               config.ffn_hidden_size,
               config.hidden_size,
               config=config,
               init_method=config.output_init_method,
               bias=False,
               tp_group=pg_collection.tp
           )

       def forward(self, x):
           # Custom MLP logic: e.g. a novel activation function or routing
           x, _ = self.linear_fc1(x)
           x = torch.sin(x)  # Example custom sine-activation MLP
           x, _ = self.linear_fc2(x)
           return x
   ```

2. Construct a `ModuleSpec` mapping these custom blocks:
   ```python
   from megatron.core.transformer.spec_utils import ModuleSpec
   from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
   from megatron.core.transformer.attention import SelfAttention

   def get_custom_layer_spec(config) -> ModuleSpec:
       return ModuleSpec(
           module=TransformerLayer,
           submodules=TransformerLayerSubmodules(
               input_layernorm=...,
               self_attention=ModuleSpec(
                   module=SelfAttention,
                   ...
               ),
               pre_mlp_layernorm=...,
               mlp=CustomParallelMLP,  # Inject your custom MLP layer
           )
       )
   ```

### Approach B: Defining a Complete Custom `LanguageModule`
If your architecture does not fit standard block sequences (e.g., state-space models like Mamba combined with attention, or non-transformer designs):
```python
from megatron.core.models.common.language_module.language_module import LanguageModule

class CustomDistributedModel(LanguageModule):
    def __init__(self, config, pre_process=True, post_process=True, **kwargs):
        super().__init__(config=config)
        self.pre_process = pre_process
        self.post_process = post_process
        # Instantiate your custom embedding layers and parallel layers manually
        ...
        
    def forward(self, input_ids, position_ids, attention_mask, **kwargs):
        # Your custom distributed forward pass
        ...
```

---

## Phase 3: Implementing the NeMo Megatron-Bridge

To enable Megatron-Bridge to understand how to translate your Hugging Face configuration and load/save weights correctly, you must write a **Bridge class**.

1. Create a bridge file: `src/megatron/bridge/models/custom/custom_bridge.py`.
2. Inherit from `MegatronModelBridge` and register it with the `@register_bridge` decorator.

```python
import torch
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.param_mapping import AutoMapping, QKVMapping, GatedMLPMapping
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM

# Import or register your Hugging Face CausalLM class
from .modeling_custom import CustomModelForCausalLM

@MegatronModelBridge.register_bridge(
    source=CustomModelForCausalLM,          # The HF Model Class
    target=GPTModel,                       # The Target Megatron Model Class (Approach A)
    model_type="custom_transformer",       # Matches the model_type in your HF config.json
)
class CustomModelBridge(MegatronModelBridge):

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> GPTModelProvider:
        """
        Maps HF config attributes to Megatron model provider parameters.
        The base class automatically handles standard parameters using CONFIG_MAPPING
        (e.g., hidden_size, layers, vocab_size, etc.).
        """
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        # 1. Map general transformer properties
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = False
        provider.position_embedding_type = "rope"
        provider.add_bias_linear = False
        
        # 2. Map custom configurations specific to your model
        # You can access self-defined properties from hf_config:
        provider.my_custom_feature = getattr(hf_config, "my_custom_feature_param", 42)

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """
        Defines the exact parameter mapping registry between HuggingFace keys
        and Megatron's tensor-parallel state dict keys.
        """
        return MegatronMappingRegistry(
            # 1. Embeddings (AutoMapping maps standard weight tensors)
            AutoMapping(
                megatron_param="embedding.word_embeddings.weight",
                hf_param="model.embed_tokens.weight",
            ),
            # 2. Output Projection Layer
            AutoMapping(
                megatron_param="output_layer.weight",
                hf_param="lm_head.weight",
            ),
            # 3. Final normalization Layer
            AutoMapping(
                megatron_param="decoder.final_layernorm.weight",
                hf_param="model.norm.weight",
            ),
            # 4. Attention QKV Fused projection (Combines separate Q, K, V projections into 1 Megatron tensor)
            QKVMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.layers.*.self_attn.q_proj.weight",
                k="model.layers.*.self_attn.k_proj.weight",
                v="model.layers.*.self_attn.v_proj.weight",
            ),
            # 5. Attention Output Projection
            AutoMapping(
                megatron_param="decoder.layers.*.self_attention.linear_proj.weight",
                hf_param="model.layers.*.self_attn.o_proj.weight",
            ),
            # 6. MLP Layer norms
            AutoMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
                hf_param="model.layers.*.input_layernorm.weight",
            ),
            AutoMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
                hf_param="model.layers.*.post_attention_layernorm.weight",
            ),
            # 7. MLP Projections (fc1/fc2)
            AutoMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                hf_param="model.layers.*.mlp.up_proj.weight",
            ),
            AutoMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc2.weight",
                hf_param="model.layers.*.mlp.down_proj.weight",
            ),
        )
```

Expose your custom bridge in `src/megatron/bridge/models/__init__.py`:
```python
from megatron.bridge.models.custom.custom_bridge import CustomModelBridge
```

---

## Phase 4: Training with Megatron-Bridge

To train your custom architecture, you will create a training recipe that specifies the layout, optimizer details, and dataloaders.

1. **Define a recipe** (`src/megatron/bridge/recipes/custom/custom_recipe.py`):
   ```python
   from megatron.bridge import AutoBridge
   from megatron.bridge.recipes.utils.config import ConfigContainer, _sft_common
   from megatron.bridge.training.gpt_step import forward_step
   from megatron.bridge.training.pretrain import pretrain

   def custom_model_pretrain_config() -> ConfigContainer:
       cfg = _sft_common()

       # 1. Setup model provider using AutoBridge from HF config/checkpoint
       cfg.model = AutoBridge.from_hf_pretrained("path/to/my/hf/model_or_config").to_megatron_provider(load_weights=False)

       # 2. Setup Parallelism config
       cfg.model.tensor_model_parallel_size = 2
       cfg.model.pipeline_model_parallel_size = 1
       cfg.model.sequence_parallel = True

       # 3. Setup Dataset
       cfg.dataset.dataset_type = "mock"  # or path to real preprocessed Megatron bin/idx
       cfg.training.max_steps = 1000
       cfg.optimizer.lr = 2e-5

       return cfg
   ```

2. **Launch the Training Job**:
   Launch using `torchrun` pointing to the training script runner with your recipe configuration:
   ```bash
   uv run python -m torch.distributed.run \
       --nproc-per-node=2 \
       scripts/training/run_recipe.py \
       --recipe custom_model_pretrain_config
   ```

---

## Phase 5: Exporting back to Hugging Face Checkpoint

Once training is complete, Megatron-LM saves sharded checkpoints on disk. Megatron-Bridge makes it simple to merge and translate these back into a Hugging Face format.

### Option A: Using the CLI (Recommended)
Megatron-Bridge includes a script `convert_checkpoints.py` that handles translation bidirectionally:

```bash
# Export the Megatron checkpoint (containing sharded weights) to HF format
uv run python examples/conversion/convert_checkpoints.py export \
  --hf-model path/to/my/hf/model_or_config \
  --megatron-path ./checkpoints/my_custom_model_mcore/iter_0001000 \
  --hf-path ./exports/my_custom_model_hf_final \
  --trust-remote-code
```

### Option B: Programmatic Weight Export
You can load the sharded weights directly into memory and stream/save them:

```python
from megatron.bridge import AutoBridge

# 1. AutoBridge automatically resolves config matching
bridge = AutoBridge.from_auto_config(
    megatron_path="./checkpoints/my_custom_model_mcore/iter_0001000",
    hf_model_id="path/to/my/hf/model_or_config",
    trust_remote_code=True
)

# 2. Run export process
bridge.export_ckpt(
    megatron_path="./checkpoints/my_custom_model_mcore/iter_0001000",
    hf_path="./exports/my_custom_model_hf_final",
    show_progress=True,
    strict=True
)
```

The resulting folder `./exports/my_custom_model_hf_final` contains:
- `config.json` (Hugging Face format)
- `model.safetensors` (Consolidated non-sharded model weights)
- Tokenizer files and supporting scripts.

You can now load and publish the model directly via the Hugging Face API:
```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./exports/my_custom_model_hf_final")
```

---

## Exporting Native MCore Models (No HF source needed for training)

If you prefer to define and train your model natively inside Megatron Core (MCore) using standard MCore training scripts (without writing any Hugging Face configuration/model classes initially), and only want the **Export to HF** functionality at the end:

1. **Decide on the target HF class**:
   Decide which Hugging Face class your MCore weights should map to. You can map them to:
   - A standard pre-existing Hugging Face class (e.g. `LlamaForCausalLM` or `GPT2LMHeadModel`), OR
   - A custom Hugging Face model class that you define (e.g. inheriting from `PreTrainedModel`).

2. **Register the Bridge between the Native MCore Model and the HF target class**:
   Create your bridge subclass registering your native MCore class as the `target` and the HF class as the `source`:
   ```python
   # src/megatron/bridge/models/my_native_mcore/my_bridge.py
   from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
   from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
   from megatron.bridge.models.conversion.param_mapping import AutoMapping

   # MyMCoreModel is your native MCore PyTorch module
   # MyHFModelClass is the target Hugging Face class (e.g. LlamaForCausalLM)
   @MegatronModelBridge.register_bridge(
       source=MyHFModelClass,
       target=MyMCoreModel,
       model_type="my_native_mcore",
   )
   class MyNativeMCoreBridge(MegatronModelBridge):
       def mapping_registry(self) -> MegatronMappingRegistry:
           # Define translation of weight keys from native MCore names to HF names
           return MegatronMappingRegistry(
               AutoMapping("embedding.word_embeddings.weight", "model.embed_tokens.weight"),
               ...
           )
   ```

3. **Train natively in MCore**:
   Train your model using standard MCore scripts (e.g., `pretrain_gpt.py`). Save the standard sharded checkpoints (which output directory paths like `checkpoints/iter_0005000/` containing the distributed shards and `run_config.yaml`).

4. **Initialize a config-only Bridge and run export**:
   When training is complete, you can export the native Megatron checkpoint directly. Since you did not initialize the model from an HF checkpoint, you instantiate the bridge as **config-only** using the target HF configuration, and call the export function:
   ```python
   from megatron.bridge import AutoBridge
   from transformers import AutoConfig
   from .my_bridge import MyNativeMCoreBridge  # Import to register

   # 1. Instantiate the target HF Configuration in memory (or load a reference template)
   hf_config = AutoConfig.for_model("my_native_mcore", vocab_size=32000, hidden_size=2048, ...)

   # 2. Build the bridge as a config-only instance
   bridge = AutoBridge.from_hf_config(hf_config)

   # 3. Export weights from the sharded native MCore checkpoint path to consolidated HF format
   bridge.export_ckpt(
       megatron_path="./checkpoints/my_mcore_model/iter_0005000",
       hf_path="./exports/my_model_hf",
       strict=True,
   )
   ```

Using this approach:
- You do **not** need Hugging Face modeling files to load/initialize weights during training.
- Training is done purely in native MCore using MCore config scripts.
- The `AutoBridge` uses your registered weight mapping to assemble and save the standard consolidated Hugging Face safetensors at the end.

