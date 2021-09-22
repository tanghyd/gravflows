# While this code works effectively, CVMFS distributed file system may be better.
# Below is a response from Jonah Kanner about best practices for data download from GWOSC.

### EMAIL

# Hi Daniel,

# Thanks for reaching out.

# In general, for large file downloads, we recommend using CVMFS:
# https://www.gw-openscience.org/cvmfs/ https://computing.docs.ligo.org/guide/cvmfs/

# The CVMFS system should be pretty robust, and should handle requests for large amounts of data well.
# I would encourage you to try it before building an in-house system.

# To instead download directly from our sever, we ask that you limit the rate of requests.
# We haven't tested this extensively, but we do know that high rates of requests can bog down the server.
# #After checking our apache configuration, we think it will handle around 25 simultaneous connections OK,
# as long as each process waits for a download to complete before making the next request.
# Actually, if you try this, we would welcome your feedback to hear about the performance.

# Thank you for your help. Good luck with your project!
# Best, Jonah Kanner

### INFO ON RUNNING SCRIPT

# dataset needs to be specified (i.e. O1, O1_16KHz, O2) etc
# 4KHz data should download much quicker than 16KHz (as expected)
# if dataset is changed between observing runs
# the gps times also need to be edited as well.

# nohup can be used for asynchronous data download in background
# bash command:
# python async_download.py \
# --queue_rate 0.1 \
# --ncon 10 \
# --buffer_size 4096 \
# --dir /mnt/datahole/daniel/gwosc  # replace with desired directory

import os
import json
import datetime
import time
import logging
import requests

import asyncio
import aiohttp

from tqdm import tqdm
from typing import Optional, Union, List, Dict
from pathlib import Path

logging.basicConfig(
    filename=f'download_logs/{time.time()}.log',
    filemode='a',
    level=logging.DEBUG,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)

# logging.getLogger('asyncio').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

async def on_request_start(session, context, params):
    "Debugging function to add a traceback object to an aiohttp.ClientSession ."
    logging.getLogger('aiohttp.client').debug(f'Starting request <{params}>')

def get_filepath_from_metadata(
    directory: Union[str, bytes, os.PathLike], 
    strain_metadata: Dict[str, Union[str, int, float]]
) -> Path:
    detector = strain_metadata['detector']
    gps_start_time = strain_metadata['GPSstart']
    gps_end_time = str(int(gps_start_time) + int(strain_metadata['duration']))
    filename = f'{detector}_{gps_start_time}-{gps_end_time}.hdf'
    return Path(directory) / strain_metadata['detector'] / filename

def get_strain_metadata(
    observation_run: str,
    gps_start_time: str,
    gps_end_time: str,
    detector: Union[str, List[str]],
    threshold: float,
    destination: Union[str, bytes, os.PathLike], 
    subset: Optional[int]=None,
):
    """Synchronous function to get GWOSC observation run metadata either
    from disk or by GET request."""
    assert 0 <= threshold <= 1, "threshold must be between 0 and 1"
    
    # get strain metadata to inform strain data download from gwosc urls
    data_dir = Path(destination) / observation_run
    metadata_path = data_dir / f'strain_{detector}_metadata.json'
    
    if os.path.isfile(metadata_path):
        # load metadata from disk if already downloaded
        with open(metadata_path, "r") as json_file:
            metadata = json.load(json_file)
        logger.debug(f'strain_{detector}_metadata loaded from disk.')
    else:
        # download metadata from gwosc url
        host = 'https://gw-openscience.org/archive/links'
        url = f'{host}/{observation_run}/{detector}/{gps_start_time}/{gps_end_time}/json/'
        response = requests.get(url)  # TO-DO: handle 404 errors etc
        metadata = response.json()
        
        # write metadata to disk for future runs
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)
        
        with open(metadata_path, "wb") as json_file:
            json_file.write(response.content)

        logger.debug(f'strain_{detector}_metadata retrieved via get request.')
      
    # extract .hdf5 strain files with sufficient % of valid data  
    metadata['strain'] = [
        strain for strain in metadata['strain']
        if strain['url'].endswith('hdf5')
        and strain['duty_cycle'] >= threshold
    ]
    
    # subset data for testing purposes
    if subset:
        metadata['strain'] = metadata['strain'][:subset]

    return metadata

async def download_strain(
    session: aiohttp.ClientSession,
    strain_metadata: Dict[str, Union[str, float, int]],
    directory: Union[str, bytes, os.PathLike],
    id: Optional[str]=None,
    buffer_size: Optional[int]=None,
) -> None:
    name = f'Writer {id}' if type(id) in [int, str] else 'Writer'  # for logger
    filepath = get_filepath_from_metadata(directory, strain_metadata)

    async with session.get(strain_metadata['url']) as response:
        assert response.status == 200, f"Writer: ERROR RESPONSE {response.status}"
        if filepath.exists(): logger.debug(f'{name}: overwriting <{filepath}>.')

        with open(filepath, mode='wb') as f: 
            start = time.perf_counter()
            logger.debug(f'{name}: opened file to write <{filepath}>')

            # stream response content and write chunked bytes to disk
            if buffer_size:
                # manually control chunk size (e.g. buffer_size=4096)
                assert 0 < buffer_size <= 2**16
                stream = response.content.iter_chunked(buffer_size)
            else:
                # get any chunk size ready from session response
                stream = response.content.iter_any()
                
            async for chunk in stream:
                f.write(chunk)

            elapsed = time.perf_counter() - start
            logger.debug(f'{name}: exiting <{filepath}> write loop after {elapsed:0.5f}s')


async def consume(
    session: aiohttp.ClientSession,
    queue: asyncio.Queue,
    directory: Union[str, bytes, os.PathLike],  # Path
    id: Optional[Union[id,str]]=None,
    buffer_size: Optional[int]=None,
    pbar: Optional[tqdm]=None,
) -> None:
    name = f'Consumer {id}' if type(id) in [int, str] else 'Consumer'  # for logger
    while True:
        # indefinitely await available data from queue
        item, start, idx = await queue.get()
        now = time.perf_counter()
        logger.debug(f'{name}: retrieved item {idx}:<{item["url"]}>; idle for {now - start:0.5f}s')

        try:
            await download_strain(session, item, directory, id, buffer_size)
        except Exception as e:
            # delete file and add back to queue
            filepath = get_filepath_from_metadata(directory, strain_metadata=item)
            filepath.unlink(missing_ok=True)  # delete if error but file exists

            # TO DO: Check aiohttp exceptions https://stackoverflow.com/a/55854094/15808586
            if pbar: pbar.set_postfix({'exceptions': 1+int(pbar.postfix.split('=',1)[1])})
            logger.debug(f'{name}: exception occured for item {idx}:<{item["url"]}>')
            logger.debug(f'{name}: <{e}>.')

            await queue.put((item, time.perf_counter(), idx)) 
        else:
            queue.task_done()
            if pbar: pbar.update(1)
            logger.debug(f'{name}: downloaded item {idx}:<{item["url"]}>.')


async def produce(
    queue: asyncio.Queue,
    data: Dict[str, Union[float, str, int]],
    data_dir: Union[str, bytes, os.PathLike],
    queue_rate: Optional[float]=None,
    skip: bool=True,
    pbar: Optional[tqdm]=None,
) -> None:
    for idx, item in enumerate(data):
        # add data and insertion times to queue
        start = time.perf_counter()  # pass start time to queue
        if skip and get_filepath_from_metadata(directory=data_dir, strain_metadata=item).exists():
            logger.debug(f'Producer: skipping item {idx}:<{item["url"]}> as file already exists.')
            if pbar: pbar.update(1)
        else:
            await queue.put((item, start, idx))  # await required if queue_length != 0
            logger.debug(f'Producer: added item {idx}:<{item["url"]}> to queue [{idx}/{len(data)}].')
            
            if queue_rate:
                # rate limit only if queue rate (expected requests per second) is not None
                assert queue_rate > 0, "queue_rate must be a positive float (seconds)."
                await asyncio.sleep(1 / queue_rate)

async def main(
    dataset: str,
    start: str,
    end: str,
    detectors: Union[str, List[str]],
    directory: Union[str, bytes, os.PathLike]=Path('gwosc'),
    threshold: float=0,
    queue_rate: Optional[float]=1,
    queue_size: int=0,
    ncon: int=1,
    buffer_size: Optional[int]=None,
    overwrite: bool=False,
    debug: bool=False,
):
    """Function downloads data from GWOSC Data Archive using an asynchronous producer-consumer architecture.

    Metadata associated with a given observation run dataset is first (synchronously) downloaded and saved.
    A producer puts metadata (i.e. urls to download strain .hdf files) into a queue shared with a pool
    of consumers. These consumer(s) are specifically handled as an aiohttp.ClientSession with a maximum
    number of concurrent connections (i.e. the number of "consumers"). All consumers will greedily .get()
    urls from the queue and open a corresponding .hdf file to write data - the response is streamed via
    the aiohttp.ClientSession object and written to disk every 1024 bytes.

    The benefit of the producer-consumer with a shared queue in this case is that we have to key parameters
    we would like to control: the number of requests per second, and the number of concurrent TCP connections.

    """
    # check input args and specify directories
    directory = Path(directory)
    data_dir = directory / dataset
    detectors = [detectors] if isinstance(detectors, str) else detectors
    assert detectors, "No detectors provided - valid interferometer keys may include: ['H1', 'L1', 'V1']."
    assert queue_size >= 0, "queue_size should be a non-negative integer."

    # get strain metadata for every detector from gwosc or disk (synchronous small file download)
    metadata = None
    metadata_pbar_desc = f'Obtaining {detectors} metadata for {dataset} run'
    metadata_pbar = tqdm(detectors, desc=metadata_pbar_desc, ncols=120) if not debug else detectors
    for detector in metadata_pbar:
        (data_dir / detector).mkdir(parents=True, exist_ok=True)  # ensure sub-directories exist

        if metadata is None:
            metadata = get_strain_metadata(dataset, start, end, detector, threshold, directory)
        else:
            strain_metadata = get_strain_metadata(dataset, start, end, detector, threshold, directory)
            for key in ['dataset', 'GPSstart', 'GPSend']:  # assumes three keys only (+ "strain")!
                assert strain_metadata[key] == metadata[key], f"{key} does not match between datasets."
            metadata['strain'].extend(strain_metadata['strain'])

    # prepare aiohttp session connectors
    connector = aiohttp.TCPConnector(limit=ncon, force_close=True)  # https://stackoverflow.com/a/67641667/15808586
    timeout = aiohttp.ClientTimeout( 
        total=None,  # default value is 5 (minutes), set to `None` for unlimited timeout
        sock_connect=300,  # how long to wait before an open socket allowed to connect
        sock_read=300+(ncon/queue_rate)  # how long to wait to read data before timing out
    )
    session_args = dict(connector=connector, timeout=timeout, raise_for_status=True)

    if not debug:
        pbar = tqdm(
            desc=f'Downloading {dataset} files',
            total=len(metadata['strain']),
            ncols=120,
            postfix={'exceptions': 0}
        )
    else:
        # add log whenever a get request is sent to a url
        trace_config = aiohttp.TraceConfig()
        trace_config.on_request_start.append(on_request_start)
        session_args['trace_configs'] = [trace_config]
        pbar = None  # no tqdm progress bar with logger statements
    
    # run download via aiohttp client session
    async with aiohttp.ClientSession(**session_args) as session:
        # producer (rate limited) >> queue (urls) >> consumers (concurrent download)
        queue = asyncio.Queue(maxsize=queue_size)
        producer = asyncio.create_task(
            produce(queue, metadata['strain'], data_dir, queue_rate, not overwrite, pbar)
        )
        consumers = [
            asyncio.create_task(
                consume(session, queue, data_dir, id, buffer_size, pbar)
            ) for id in range(ncon)
        ]

        await asyncio.gather(producer)  # do not proceed until all urls put in queue
        await queue.join()  # triggered when queue is empty (all downloads complete)
        for consumer in consumers:
            # exit while loop in consumer coroutine
            consumer.cancel()  

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parse arguments for asynchronous GWOSC download script.")
    int_or_none = lambda b: None if b in ['None', None] else int(b)
    parser.add_argument("-d", "--dataset", type=str, default='O1_16KHZ', help='LIGO-Virgo Collaboration observation run (e.g. O1, O2, O3a).')
    parser.add_argument("-s", "--start", type=str, default='1126051217', help='GPS start time for strain data.')
    parser.add_argument("-e", "--end", type=str, default='1137254417', help='GPS end time for strain data.')
    parser.add_argument("-i", "--detectors", type=str, default=['H1', 'L1'], nargs='+', help='LIGO-Virgo Collaboration interferometers (e.g. H1, L1, V1).')
    parser.add_argument("-t", "--threshold", type=float, default=0., help='Threshold for percentage of valid data required for data download.')
    parser.add_argument("-n", "--ncon", type=int, default=1, help='Number of concurrent consumers for download session.')
    parser.add_argument("-r", "--queue_rate", type=float, default=1., help='URL generation rate for producer in item/sec.')
    parser.add_argument("-q", "--queue_size", type=int, default=0, help='Maximum size of shared queue for producer-consumer coroutines.')
    parser.add_argument("-b", "--buffer_size", type=int_or_none, nargs='?', default=None, help='Minimum chunk size before writing streamed response (in bytes).')
    parser.add_argument("-o", "--overwrite", action="store_true", default=False, help='Whether to overwrite data files if they already exist on disk.')
    parser.add_argument("-dir", "--directory", type=str, default='gwosc', help='Output data directory for writing files to disk.')
    parser.add_argument("-db", "--debug", default=False, action="store_true", help="Whether to enable asyncio.run debug mode.")
    args = parser.parse_args()
    logger.debug(f"Arguments: {args.__dict__}")

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger('asyncio').setLevel(logging.DEBUG)
        logging.getLogger('aiohttp.client').setLevel(logging.DEBUG)

    # run async download routine
    start = time.perf_counter()

    asyncio.run(main(**args.__dict__), debug=args.debug)

    elapsed = datetime.timedelta(seconds=time.perf_counter() - start)
    logger.debug(f"Arguments: {args.__dict__}")
    logger.debug(f"Program completed in {elapsed}.")