import evaluation
import experiment
from experiments.vae_k_1_layers_2 import experiment as vae_k_1_layers_2
from experiments.vae_k_1_layers_1 import experiment as vae_k_1_layers_1
from experiments.iwae_k_50_layers_1_retrained_as_vae_k_1 import experiment as iwae_k_50_layers_1_retrained_as_vae_k_1
import utils

one_layer = vae_k_1_layers_1
one_layer['data']['batch_size'] = 100
one_layer['data']['num_workers'] = 0

two_layer = vae_k_1_layers_2
data_loader, _, bias = utils.setup_data(one_layer["data"])
data_test = data_loader['test']
model_1, _ = utils.setup_model(one_layer["model"], bias)
model_2, _ = utils.setup_model(two_layer["model"], bias)
NAMES_CHECKPOINTS = [
    # PREVIOUSLY computed:
    ('vae_k_1_layers_1_epoch_1500', 'results/for_evaluation/vae_k_1_layers_1_bs_20_Epoch--1500-Loss--0.00-NLL_k--0.00.pt'),
    ('vae_k_50_layers_2_epoch_1600', 'results/vae_k_50_layers_2/08-01-2022-11--42--02/Epoch--1600-Loss--0.00-NLL_k--0.00.pt'),
    ('vae_k_5_layers_2', 'results/vae_k_5_layers_2/08-01-2022-11--44--25/Epoch--1600-Loss--0.00-NLL_k--0.00.pt'),

    # BEING computed on the VM:
    ('iwae_k_50_layers_1', 'results/iwae_k_50_layers_1/09-01-2022-12--11--26/Epoch--600-Loss--0.00-NLL_k--0.00.pt'),
    ('vae_k_50_layers_1', 'results/vae_k_50_layers_1/09-01-2022-12--21--38/Epoch--600-Loss--0.00-NLL_k--0.00.pt'),
    ('vae_k_5_layers_1', 'results/vae_k_5_layers_1/09-01-2022-12--22--21/Epoch--600-Loss--0.00-NLL_k--0.00.pt'),
    ('vae_k_1_layers_2', 'results/vae_k_1_layers_2/09-01-2022-12--20--31/Epoch--300-Loss--0.00-NLL_k--0.00.pt'),
    ('iwae_k_5_layers_2', 'results/iwae_k_5_layers_2/09-01-2022-12--34--08/Epoch--300-Loss--0.00-NLL_k--0.00.pt'),
    ('iwae_k_50_layers_2', 'results/iwae_k_50_layers_2/09-01-2022-12--15--05/Epoch--300-Loss--0.00-NLL_k--0.00.pt'),
    ('iwae_k_5_layers_1', 'results/iwae_k_5_layers_1/09-01-2022-12--38--23/Epoch--150-Loss--0.00-NLL_k--0.00.pt'),
]

    # experiment.load(model, 'vae_k_50_1_layers_1_chkp.pt')
print(f"{'model': ^20}|{'nll': ^20}|{'active_units': ^20}")
for name, chkpt_path in NAMES_CHECKPOINTS:
    is_one_layer = 'layers_1' in name 
    model = model_1 if is_one_layer else model_2
    experiment.load(model, chkpt_path)
    nll = evaluation.measure_estimated_log_likelihood(data_test, model,
                                                      num_samples=5000,
                                                      num_points=500)
    z_variances = utils.get_units_variances(model, data_loader)
    active_units = str(utils.num_active_units(z_variances))
    print(f"{name: ^20}|{nll: ^20}|{active_units: ^20}")
