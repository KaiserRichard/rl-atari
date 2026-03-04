'''
Utility: Exploration Schedule
'''

def linear_schedule(start: float, end: float, duration: int, step: int):
    ''''
    Linearly decays a value over time

    Fomula: 
        value = start + slope * step
    
    Ensures value never goes below end
    '''

    # Current training step > number of decay steps
    if step >= duration: return end

    slope = (end - start) / duration
    return start + slope * step