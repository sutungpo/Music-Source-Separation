import subprocess


inference_base = '.\env\python.exe inference.py'


def determine_module_inputs(if_sep_vocal, if_sep_kara, if_sep_reverb, if_denoise):
    input_dir = "input"
    sep_output_dir = "separation_results"
    karaoke_output_dir = "karaoke_results"
    reverb_output_dir = "deverb_results"
    denoise_output_dir = "denoise_results"

    module_inputs = {
        "sep_vocal_input": None,
        "karaoke_input": None,
        "reverb_input": None,
        "denoise_input": None
    }

    previous_output_dir = input_dir

    if if_sep_vocal == 'y':
        module_inputs["sep_vocal_input"] = previous_output_dir
        previous_output_dir = sep_output_dir

    if if_sep_kara == 'y':
        module_inputs["karaoke_input"] = previous_output_dir
        previous_output_dir = karaoke_output_dir

    if if_sep_reverb == 'y':
        module_inputs["reverb_input"] = previous_output_dir
        previous_output_dir = reverb_output_dir

    if if_denoise == 'y':
        module_inputs["denoise_input"] = previous_output_dir

    return module_inputs


print("\n============")
print("分离人声选项:")
print("============\n")

if_sep_vocal = input("是否进行人声分离（y/n）\n：")

if if_sep_vocal == 'y':
    vocal_model_name = input(
        "\n请选择需要使用的模型:\n0为使用bs-roformer-1297\n1为使用bs-roformer-1296（注：1297的SDR稍高，但有反馈指出可能引入极高频上的噪音，1296则没有）\n2为使用Kim_mel_band_roformer，SDR略微好于前两个且用时减半\n：")
    if vocal_model_name == '0':
        vocal_model_name = 'model_bs_roformer_ep_317_sdr_12.9755.ckpt'
    if vocal_model_name == '1':
        vocal_model_name = 'model_bs_roformer_ep_368_sdr_12.9628.ckpt'
    if vocal_model_name == '2':
        vocal_model_name = 'MelBandRoformer_kim.ckpt'

print("\n============")
print("分离和声选项:")
print("也可以单独使用来分离带和声的伴奏")
print("============\n")

if_sep_kara = input("是否进行和声分离（y/n）\n(注：目前仅有aufr33与viperx的mel_band_roformer_karaoke一个可选，激进，但性能比UVR现有模型都要好非常多)\n：")

if if_sep_kara == 'y':
    kara_model_name = 'mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt'

print("\n============")
print("分离混响和声选项:")
print("请注意暂无法单独去混响，和声也可能有残留，需要去和声建议同时启用去和声模块")
print("============\n")

if_sep_reverb = input("是否进行和声混响分离（y/n）\n：")

if if_sep_reverb == 'y':
    reverb_model_name = input(
        "\n请选择需要使用的模型:\n  0 为使用最初版mel_band_roformer，在去混响和和声上有不错的平衡\n  1 为使用bs_roformer_8_384_10，推荐，使用了更多数据与新的增强方式训练的新bs模型,\n    SDR比2要高不少，但在和声分离上比mel系的要保守\n  2 为使用bs_roformer_8_256_8，旧的bs模型\n  3 为使用mel_band_roformer_8_256_6，非常激进的去混响，视曲目不同可能会把人声剥残，但更激进也在一些场合有更好的效果\n  4 为使用mel_band_roformer_8_512_12，比2更大的网络带来稍高的SDR的同时消耗3倍以上推理时间\n：")
    if reverb_model_name == '0':
        reverb_model_name = 'deverb_mel_band_roformer_ep_27_sdr_10.4567.ckpt'
    if reverb_model_name == '1':
        reverb_model_name = 'deverb_bs_roformer_8_384dim_10depth.ckpt'
    if reverb_model_name == '2':
        reverb_model_name = 'deverb_bs_roformer_8_256dim_8depth.ckpt'
    if reverb_model_name == '3':
        reverb_model_name = 'deverb_mel_band_roformer_8_256dim_6depth.ckpt'
    if reverb_model_name == '4':
        reverb_model_name = 'deverb_mel_band_roformer_8_512dim_12depth.ckpt'

print("\n============")
print("降噪选项:")
print("============\n")

if_denoise = input("是否进行降噪（y/n）\n：")

if if_denoise == 'y':
    denoise_model_name = input(
        "\n请选择需要使用的模型:\n0为使用SDR 27.9959的通常版模型\n1为使用SDR 27.9768的激进版模型\n：")
    if denoise_model_name == '0':
        denoise_model_name = 'denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt'
    if denoise_model_name == '1':
        denoise_model_name = 'denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt'


if_fast = input("\n是否使用快速推理模式（y/n）\n（注：开启将以更低的推理参数进行推理，以SDR稍低的代价换来数倍的速度提升，适合配置不高或者大批量推理的场合）\n：")

module_inputs = determine_module_inputs(if_sep_vocal, if_sep_kara, if_sep_reverb, if_denoise)


print("\n============")
print("推理配置清单")
print("============\n")

if if_sep_vocal == 'y':
    print(f"分离人声：已启用")
    print(f"使用的模型：{vocal_model_name}")
    print(f"输入目录：{module_inputs['sep_vocal_input']}")
    print(f"输出目录：separation_results\n")
else:
    print(f"分离人声：已禁用\n")

if if_sep_kara == 'y':
    print(f"分离和声：已启用")
    print(f"使用的模型：{kara_model_name}")
    print(f"输入目录：{module_inputs['karaoke_input']}")
    print(f"输出目录：karaoke_results\n")
else:
    print(f"分离和声：已禁用\n")

if if_sep_reverb == 'y':
    print(f"分离混响和声：已启用")
    print(f"使用的模型：{reverb_model_name}")
    print(f"输入目录：{module_inputs['reverb_input']}")
    print(f"输出目录：deverb_results\n")
else:
    print(f"分离和声混响：已禁用\n")

if if_denoise == 'y':
    print(f"降噪：已启用")
    print(f"使用的模型：{denoise_model_name}")
    print(f"输入目录：{module_inputs['denoise_input']}")
    print(f"输出目录：denoise_results\n")
else:
    print(f"降噪：已禁用\n")

if if_fast == 'y':
    print(f"快速推理模式：已启用\n")
else:
    print(f"快速推理模式：已禁用\n")

check_menu = input("按回车键继续")


if if_sep_vocal == 'y':
    print('================开始分离人声================')
    inference_case = inference_base  + f' --start_check_point pretrain/{vocal_model_name}' + f' --input_folder {module_inputs["sep_vocal_input"]} --store_dir separation_results --extract_instrumental'
    if vocal_model_name == 'model_bs_roformer_ep_317_sdr_12.9755.ckpt':
        if if_fast == 'y':
            inference_case = inference_case + ' --config_path configs/model_bs_roformer_ep_317_sdr_12.9755-fast.yaml' + ' --model_type bs_roformer'
        else:
            inference_case = inference_case + ' --config_path configs/model_bs_roformer_ep_317_sdr_12.9755.yaml' + ' --model_type bs_roformer'

    if vocal_model_name == 'model_bs_roformer_ep_368_sdr_12.9628.ckpt':
        if if_fast == 'y':
            inference_case = inference_case + ' --config_path configs/model_bs_roformer_ep_368_sdr_12.9628-fast.yaml' + ' --model_type bs_roformer'
        else:
            inference_case = inference_case + ' --config_path configs/model_bs_roformer_ep_368_sdr_12.9628.yaml' + ' --model_type bs_roformer'

    if vocal_model_name == 'MelBandRoformer_kim.ckpt':
        if if_fast == 'y':
            inference_case = inference_case + ' --config_path configs/config_vocals_mel_band_roformer_kim-fast.yaml' + ' --model_type mel_band_roformer'
        else:
            inference_case = inference_case + ' --config_path configs/config_vocals_mel_band_roformer_kim.yaml' + ' --model_type mel_band_roformer'
    subprocess.run(f'{inference_case}', shell=True)


if if_sep_kara == 'y':
    print('================开始分离和声================')
    inference_case = inference_base + ' --model_type mel_band_roformer' + f' --start_check_point pretrain/{kara_model_name}' + f' --input_folder {module_inputs["karaoke_input"]} --store_dir karaoke_results --extract_karaoke'
    if if_fast == 'y':
        inference_case = inference_case + ' --config_path configs/config_mel_band_roformer_karaoke-fast.yaml'
    else:
        inference_case = inference_case + ' --config_path configs/config_mel_band_roformer_karaoke.yaml'


    subprocess.run(f'{inference_case}', shell=True)


if if_sep_reverb == 'y':
    print('================开始分离混响和声================')
    inference_case = inference_base + f' --start_check_point pretrain/{reverb_model_name}' + f' --input_folder {module_inputs["reverb_input"]} --store_dir deverb_results --extract_reverb'
    if reverb_model_name == 'deverb_mel_band_roformer_ep_27_sdr_10.4567.ckpt':
        if if_fast == 'y':
            inference_case = inference_case + ' --config_path configs/deverb_mel_band_roformer-fast.yaml  --model_type mel_band_roformer'
        else:
            inference_case = inference_case + ' --config_path configs/deverb_mel_band_roformer.yaml  --model_type mel_band_roformer'

    if reverb_model_name == 'deverb_bs_roformer_8_384dim_10depth.ckpt':
        if if_fast == 'y':
            inference_case = inference_case + ' --config_path configs/deverb_bs_roformer_8_384dim_10depth-fast.yaml  --model_type bs_roformer'
        else:
            inference_case = inference_case + ' --config_path configs/deverb_bs_roformer_8_384dim_10depth.yaml  --model_type bs_roformer'

    if reverb_model_name == 'deverb_bs_roformer_8_256dim_8depth.ckpt':
        if if_fast == 'y':
            inference_case = inference_case + ' --config_path configs/deverb_bs_roformer_8_256dim_8depth-fast.yaml  --model_type bs_roformer'
        else:
            inference_case = inference_case + ' --config_path configs/deverb_bs_roformer_8_256dim_8depth.yaml  --model_type bs_roformer'

    if reverb_model_name == 'deverb_mel_band_roformer_8_256dim_6depth.ckpt':
        if if_fast == 'y':
            inference_case = inference_case + ' --config_path configs/8_256_6_deverb_mel_band_roformer_8_256dim_6depth-fast.yaml  --model_type mel_band_roformer'
        else:
            inference_case = inference_case + ' --config_path configs/8_256_6_deverb_mel_band_roformer_8_256dim_6depth.yaml  --model_type mel_band_roformer'

    if reverb_model_name == 'deverb_mel_band_roformer_8_512dim_12depth.ckpt':
        if if_fast == 'y':
            inference_case = inference_case + ' --config_path configs/8_512_12_deverb_mel_band_roformer_8_512dim_12depth-fast.yaml  --model_type mel_band_roformer'
        else:
            inference_case = inference_case + ' --config_path configs/8_512_12_deverb_mel_band_roformer_8_512dim_12depth.yaml  --model_type mel_band_roformer'
    subprocess.run(f'{inference_case}', shell=True)


if if_denoise == 'y':
    print('================开始降噪================')
    inference_case = inference_base + f' --model_type mel_band_roformer --start_check_point pretrain/{denoise_model_name}' + f' --input_folder {module_inputs["denoise_input"]} --store_dir denoise_results --extract_noise'
    if denoise_model_name == 'denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt':
        if if_fast == 'y':
            inference_case = inference_case + ' --config_path configs/model_mel_band_roformer_denoise-fast.yaml'
        else:
            inference_case = inference_case + ' --config_path configs/model_mel_band_roformer_denoise.yaml'

    if denoise_model_name == 'denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt':
        if if_fast == 'y':
            inference_case = inference_case + ' --config_path configs/model_mel_band_roformer_denoise-fast.yaml'
        else:
            inference_case = inference_case + ' --config_path configs/model_mel_band_roformer_denoise.yaml'
    subprocess.run(f'{inference_case}', shell=True)

print('================处理完成================')



