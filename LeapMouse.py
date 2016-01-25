import os, sys, inspect, thread, time
#import numpy
import math
import Leap
import Mouse
from Leap import CircleGesture, KeyTapGesture, ScreenTapGesture, SwipeGesture

def smoother(x): # smooth the raw data to remove any small jerks
    sign = x / abs(x)
    return (sign * (math.log(1 + math.exp(abs(x))) - math.log(2)))

class MouseListener(Leap.Listener):
    def __init__(self):
        super(MouseListener, self).__init__()  #Initialize like a normal listener
        #Initialize a bunch of stuff specific to this implementation
        self.cursor = Mouse.relative_cursor()  #The cursor object that lets us control mice cross-platform
        self.clicked = False
        self.velocity = 0.05

    def on_connect(self, controller):
        print "Connected"

    def on_frame(self, controller):
        frame = controller.frame()

        if not frame.hands.is_empty:  #Make sure we have some hands to work with
            rightmost_hand = max(frame.hands, key=lambda hand: hand.palm_position.x)
            cur_pos = rightmost_hand.palm_velocity
            print cur_pos, '; ', rightmost_hand.pinch_strength, '; ', rightmost_hand.grab_strength
            self.cursor.move(round(smoother(rightmost_hand.palm_velocity[0]) * self.velocity, 0), round(smoother(rightmost_hand.palm_velocity[1]) * -self.velocity, 0))

            k = 1 / (math.log(1 + math.e) - math.log(2))
            if rightmost_hand.pinch_strength >= smoother(0.95) * k:
                if self.clicked == False:
                    self.cursor.click()
                    self.clicked = True
            else:
                self.clicked = False
                if rightmost_hand.grab_strength >= smoother(1) * k:
                    self.cursor.click_down()
                    #self.velocity = 0.0125
                    self.velocity = 0.05
                elif rightmost_hand.grab_strength < smoother(1) * k:
                    self.cursor.click_up()
                    self.velocity = 0.05

def main():

    # Create a sample listener and controller
    listener = MouseListener()
    controller = Leap.Controller()

    # Have the sample listener receive events from the controller
    controller.add_listener(listener)
    # Keep this process running until Enter is pressed
    print "Press Enter to quit..."
    try:
        sys.stdin.readline()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
  main()