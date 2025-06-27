"""
Migrated test script originally from openhands.nvidia.swe_agent.utils
"""

from openhands.nvidia.swe_agent.utils import *

if __name__ == '__main__':
    dataset = pd.read_parquet(
        '/lustre/fsw/portfolios/nvr/users/mingjiel/data/swegym/train.parquet'
    )

    instance = dataset.iloc[0]['instance']
    # convert instance to serializable pandas series
    instance = pd.Series(instance)
    instance = instance.apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    # try:
    #     # Try to get the current event loop
    #     loop = asyncio.get_event_loop()
    # except RuntimeError:
    #     # No event loop exists in this thread, create a new one
    #     loop = asyncio.new_event_loop()
    #     asyncio.set_event_loop(loop)

    # result = loop.run_until_complete(run(instance))
    # print("BEGIN RESULTS.")
    # print(result)
    # print("END RESULTS.")

    mock_patch = """
    diff --git a/openhands_patch_test.txt b/openhands_patch_test.txt
    new file mode 100644
    index 0000000..e69de29
    --- /dev/null
    +++ b/openhands_patch_test.txt
    @@
    +This is an OpenHands test patch.
    """

    ###### async evaluate ##########
    async def run_parallel_async():
        tasks = []
        for idx in range(1):
            inst_clone = instance.copy()
            inst_clone['instance_id'] = f'{instance["instance_id"]}_{idx}'
            gold_patch = inst_clone['patch']
            tasks.append(evaluate_agent(gold_patch, inst_clone))
        return await asyncio.gather(*tasks, return_exceptions=True)

    try:
        all_reports = asyncio.run(run_parallel_async())
        for i, rep in enumerate(all_reports):
            print(f'\nBEGIN EVAL REPORT [{i}]')
            print(rep)
            print(f'END EVAL REPORT [{i}]')
    except Exception as eval_err:
        logger.error(f'Failed to parallel-evaluate patches: {eval_err}')
        raise

    ###### sequential evaluate (non-async) ##########
    async def run_sequential_async():
        results = []
        for i in range(2):
            res = await evaluate_agent(mock_patch, instance)
            print(f'\nBEGIN EVAL REPORT SEQ [{i}]')
            print(res)
            print(f'END EVAL REPORT SEQ [{i}]')
            results.append(res)
        return results

    try:
        asyncio.run(run_sequential_async())
    except Exception as seq_err:
        logger.error(f'Sequential evaluation failed: {seq_err}')
        raise
