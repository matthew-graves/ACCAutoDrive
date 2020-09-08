import mmap
import numpy as np
import struct
from time import sleep

class dataStorage:
    packetId = 0 # 0 - 4
    gas = 0 # 4-8
    brake = 0 # 8-12
    fuel = 0 # 12-16
    gear = 0 # 16-20
    rpms = 0 # 20-24
    steerAngle = 0 # 24-28
    speedKmh = 0 # 28-32
    velocity = np.zeros(3)
    accG = np.zeros(3)
    wheelSlip = np.zeros(4)
    wheelLoad = np.zeros(4)
    wheelsPressure = np.zeros(4)
    wheelAngularSpeed = np.zeros(4)
    tyreWear = np.zeros(4)
    tyreDirtyLevel = np.zeros(4)
    tyreCoreTemperature = np.zeros(4)
    camberRAD = np.zeros(4)
    suspensionTravel = np.zeros(4)
    drs = 0
    tc = 0
    heading = 0
    pitch = 0
    roll = 0
    cgHeight = 0
    carDamage = np.zeros(5)
    numberOfTyresOut = 0
    pitLimiterOn = 0
    abs = 0
    kersCharge = 0
    kersInput = 0
    autoShifterOn = 0
    rideHeight = np.zeros(2)
    turboBoost = 0
    ballast = 0
    airDensity = 0
    airTemp = 0
    roadTemp = 0
    localAngularVel = np.zeros(3)
    finalFF = 0
    performanceMeter = 0
    engineBrake = 0
    ersRecoveryLevel = 0
    ersPowerLevel = 0
    ersHeatCharging = 0
    ersIsCharging = 0
    kersCurrentKJ = 0
    drsAvailable = 0
    drsEnabled = 0
    brakeTemp= np.zeros(4)
    clutch = 0
    tyreTempI = np.zeros(4)
    tyreTempM = np.zeros(4)
    tyreTempO = np.zeros(4)
    isAIControlled = 0
    tyreContactPoint = np.zeros((4, 3))
    tyreContactNormal = np.zeros((4, 3))
    tyreContactHeading = np.zeros((4, 3))
    brakeBias = 0
    localVelocity = np.zeros(3)
    P2PActivations = 0
    P2PStatus = 0
    currentMaxRpm = 0
    mz = np.zeros(4)
    fx = np.zeros(4)
    fy = np.zeros(4)
    slipRatio = np.zeros(4)
    slipAngle = np.zeros(4)
    tcinAction = 0
    absInAction = 0
    suspensionDamage = np.zeros(4)
    tyreTemp = np.zeros(4)

data = dataStorage

z = bytearray()
x = 0
gas = struct
brake = struct
steer = struct
speed = struct


shm = mmap.mmap(0, 712, "Local\\acpmf_physics", access=mmap.ACCESS_READ)
if shm:
    while True:
        a = shm.readline()
        try:
            gas = struct.unpack('f', a[4:8])
            brake = struct.unpack('f', a[8:12])
            steer = struct.unpack('f', a[24:28])
            speed = struct.unpack('f', a[28:32])

            print(gas)
            print(steer)
            print(brake)
            print(speed)
        except Exception as e:
            pass

        sleep(.25)
        shm.seek(0)
        # for i, x in enumerate(a):
        #     if i > 3 and i < 8:
        #         z.append(x)
        #     if i > 8 and z != bytearray():
        #         try:
        #             print(struct.unpack('f', z))
        #             z = bytearray()
        #         except Exception as e:
        #             print('f')




