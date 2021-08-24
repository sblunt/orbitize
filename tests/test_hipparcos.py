from orbitize.hipparcos import HipparcosLogProb

def test_hipparcos_api():

    # check that error is caught for a star with solution type != 5 param
    hip_num = '25'
    num_secondary_bodies = 1
    iad_file = '/data/user/sblunt/HipIAD/H{}/HIP{}.d'.format(hip_num[0:3], hip_num)
    iad_file = '/data/user/sblunt/HipIAD/H{}/HIP{}.d'.format(hip_num[0:3], hip_num)

    try:
        myHip = HipparcosLogProb(iad_file, hip_num, num_secondary_bodies)
    except Exception: 
        pass


if __name__ == '__main__':
    test_hipparcos_api()