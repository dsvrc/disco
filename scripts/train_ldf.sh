# python train_ldf.py --config configs/aistpp_263d/ldf_aistpp_263d.yaml --override train=True
export CUDA_VISIBLE_DEVICES="4,7"
# nohup python train_ldf.py --config configs/finedance_263d/ldf_multi.yaml --override train=True > outputs/ldf_multi_train.log 2>&1 &
# nohup python train_ldf.py --config configs/finedance_263d/ldf_multi.yaml > outputs/ldf_multi_test.log 2>&1 &
# nohup python train_ldf.py --config configs/aistpp_263d/ldf_aistpp_263d_stream.yaml --override train=True > outputs/ldf_aistpp_263d_stream_train.log 2>&1 &
# python train_ldf.py --config configs/aistpp/ldf_aistpp_263d_sliced_stream.yaml --override train=True 2>&1 | tee outputs/ldf_aistpp_263d_sliced_stream_train_1.log
# python train_ldf.py --config configs/aistpp/ldf_aistpp_263d_sliced_stream_hg.yaml --override train=True 2>&1 | tee outputs/ldf_aistpp_263d_sliced_stream_hg_train_2.log
# python train_ldf.py --config configs/aistpp/ldf_aistpp_272d_sliced_stream.yaml --override train=True --override debug=False 2>&1 | tee outputs/ldf_aistpp_272d_sliced_stream_train_2.log
# python train_ldf.py --config configs/aistpp/ldf_aistpp_272d_sliced_stream_hg.yaml --override train=True --override debug=False 2>&1 | tee outputs/ldf_aistpp_272d_sliced_stream_hg_train_2.log
# python train_ldf.py --config configs/aistpp/ldf_aistpp_272d_stream.yaml --override train=True --override debug=False 2>&1 | tee outputs/ldf_aistpp_272d_stream_train.log
# python train_ldf.py --config configs/aistpp/ldf_aistpp_272d_stream_pae.yaml --override train=True --override debug=False 2>&1 | tee outputs/ldf_aistpp_272d_stream_pae_train.log
# python train_ldf.py --config configs/aistpp/ldf_aistpp_272d_stream_hg.yaml --override train=True --override debug=False 2>&1 | tee outputs/ldf_aistpp_272d_stream_hg_train.log
# python train_ldf.py --config configs/aistpp/ldf_aistpp_272d_stream_hg.yaml --override train=True --override debug=False 2>&1 | tee outputs/ldf_aistpp_272d_stream_hg_random_train.log
python train_ldf.py --config configs/aistpp/ldf_aistpp_272d_stream_pae_hg.yaml --override train=True --override debug=False 2>&1 | tee outputs/ldf_aistpp_272d_stream_pae_hg_train.log
# python train_ldf.py --config configs/aistpp/ldf_aistpp_272d_stream_pae.yaml --override train=True --override debug=False --override prediction_type="noise" 2>&1 | tee outputs/ldf_aistpp_272d_stream_pae_train.log
# python train_ldf.py --config configs/ldf_multi.yaml --override train=True --override debug=False 2>&1 | tee outputs/ldf_multi_stand_train.log
# python train_ldf.py --config configs/ldf_multi.yaml --override train=False --override debug=False 2>&1 | tee outputs/ldf_multi_stand_test.log
# python pretokenize_vae.py --config configs/vae_wan_1d_multi_272.yaml
# python train_ldf.py --config configs/ldf_multi_hg.yaml --override train=True --override debug=False 2>&1 | tee outputs/ldf_multi_stand_hg_train.log
# python train_ldf.py --config configs/ldf_multi.yaml --override train=True --override debug=False 2>&1 | tee outputs/ldf_multi_train.log
# python train_ldf.py --config configs/ldf_multi_stand.yaml --override train=True --override debug=False 2>&1 | tee outputs/ldf_multi_stand_train.log
