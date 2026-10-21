export CUDA_VISIBLE_DEVICES=5

# nohup python train_vae.py --config configs/finedance/vae_wan_1d_finedance.yaml --override train=True > outputs/vae_wan_1d_finedance_train.log 2>&1 &
# # nohup python train_vae.py --config configs/finedance/vae_wan_1d_finedance.yaml > outputs/vae_wan_1d_finedance_test.log 2>&1 &

# nohup python train_vae.py --config configs/aistpp_263d/vae_wan_1d_aistpp_272d.yaml --override train=True > outputs/vae_wan_1d_aistpp_272d_train_8_0_0.log 2>&1 &
# nohup python train_vae.py --config configs/finedance_263d/vae_wan_1d_finedance_272d.yaml --override train=True > outputs/vae_wan_1d_finedance_272d_train_8_0_0.log 2>&1 &

python train_vae.py --config configs/vae_wan_1d_multi_272.yaml --override train=True 2>&1 | tee outputs/vae_wan_1d_multi_272_train.log