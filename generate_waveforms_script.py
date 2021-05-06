import os
import argparse
import lfigw.waveform_generator as wfg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Waveform generator script.')
    parser.add_argument('-e', '--event',
        help="Gravitational Wave Event",
        type=str, default='GW150914')
    
    parser.add_argument('--outdir',
        help='Directory to save generated waveform outputs/',
        type=str, default='waveforms')
    
    parser.add_argument('--generate_reduced_basis',
        help='Number of reduced basis vectors to generate',
        type=int, default=50000)
    
    parser.add_argument('--generate_dataset',
        help='Number of data points to generate',
        type=int, default=1000000)
    
    parser.add_argument('--generate_noisy_test_data',
        help='Number of noisy test data points to generate',
        type=int, default=5000)
    
    args = parser.parse_args()

    # check valid gravitational wave event input
    gw_event = args.event
    valid_gw_events = ['GW150914', 'GW170814']
    assert gw_event in valid_gw_events, (
        f"event {gw_event} must be one of {valid_gw_events}"
    )
    
    # check save dir
    assert os.path.exists(args.outdir)
    save_path = os.path.join(args.outdir, gw_event)
    
    wfd = wfg.WaveformDataset(
        spins_aligned=False,
        domain='RB',
        extrinsic_at_train=True
    )

    wfd.Nrb = 600
    wfd.approximant = 'IMRPhenomPv2'

    wfd.load_event(f'data/events/{gw_event}/')

    wfd.importance_sampling = 'uniform_distance'

    # edit priors
    wfd.prior['distance'] = [100.0, 1000.0]
    wfd.prior['a_1'][1] = 0.88
    wfd.prior['a_2'][1] = 0.88

    print('Dataset properties')
    print('Event', wfd.event)
    print(wfd.prior)
    print('f_min', wfd.f_min)
    print('f_min_psd', wfd.f_min_psd)
    print('f_max', wfd.f_max)
    print('T', wfd.time_duration)
    print('reference time', wfd.ref_time)

    # long runtime
    wfd.generate_reduced_basis(args.generate_reduced_basis)
    wfd.generate_dataset(args.generate_dataset)
    wfd.generate_noisy_test_data(args.generate_noisy_test_data)

    # save generated waveform data
    wfd.save(save_path)
    wfd.save_train(save_path)
    wfd.save_noisy_test_data(save_path)

    print('Program complete. Waveform dataset has been saved.')
