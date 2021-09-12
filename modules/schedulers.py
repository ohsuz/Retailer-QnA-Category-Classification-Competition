import torch.optim as optim


def get_scheduler():
    """ Scheduler 반환 함수

    Args:
        params (dict): scheduler 선언에 필요한 파라미터 딕셔너리
    
    Returns:
        scheduler (`scheduler`)
    """
    # Unpack params
    PARAM_1 = param['param_1']
    PARAM_2 = param['param_2']

    # Declare scheduler
    scheduler = scheduler(PARAM_1, PARAM_2)

    return scheduler

