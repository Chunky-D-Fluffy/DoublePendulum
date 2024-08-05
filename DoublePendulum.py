import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class Pendulum:
    def __init__(self, theta1, theta2, dt):
        
        self.theta1 = theta1    #Vertical angle between connection point and mass 1 string
        self.theta2 = theta2    #Vertical angle between imaginary line drawn down from mass 1 and mass 2 string

        self.p1 = 0.0
        self.p2 = 0.0           #Assume starting initial velocities are 0

        self.dt = dt            #specified time-step, smaller will give more accurate results (numerical derivative)

        self.g = 9.81           #acceleration due to gravity (m/s^2)
        self.length = 1.0       #length of string (m)

        self.trajectory = [self.coord_conversion()]   #The simulation should be in cartesian coordinates, but we solved the equation in polar coordinates!

    def coord_conversion(self):
        '''
        This method takes the current position of the masses in terms of polar coordinates (angles and lengths)
        and converts them to Cartesian coordinates. 
        '''
        x1 = self.length * np.sin(self.theta1)
        y1 = -self.length * np.cos(self.theta1)

        x2 = x1 + self.length * np.sin(self.theta2)
        y2 = y1 - self.length * np.cos(self.theta2)

        return np.array([[0.0, 0.0], [x1,y1], [x2,y2]]) #returns the Cartesian coordinate locations of the pivot point, mass 1, and mass 2
    
    def movement(self):
        '''
        This method will be used to get the position by solving for theta 1 and theta 2 for every timestep
        using our EOM
        '''
        #Variables defintion
        theta1 = self.theta1
        theta2 = self.theta2
        p1 = self.p1
        p2 = self.p2
        g = self.g
        l = self.length

        #Variables to simplify the equations
        A = np.cos(theta1 - theta2)
        B = np.sin(theta1 - theta2)
        C = (1 + B**2)
        D = p1 * p2 * B / C
        E = (p1**2 + 2 * p2**2 - p1 * p2 * A) \
        * np.sin(2 * (theta1 - theta2)) / 2 / C**2
        F = D - E

        #Hamilton's equations
        self.theta1 += self.dt * (p1 - p2 * A) / C #our theta_dot_1 equation
        self.theta2 += self.dt * (2 * p2 - p1 * A) / C #our theta_dot_2 equation

        self.p1 += self.dt * (-2 * g * l * np.sin(theta1) - F)  #our p_dot_1 equation
        self.p2 += self.dt * (-g * l * np.sin(theta2) + F)       #our p_dot_2 equation

        #The computer can now use these 4 coupled first order ODEs to ascetain theta1 and theta2 for this time step
        position = self.coord_conversion()  #convert the theta1 and theta2 coordinates to cartesian                 
        
        self.trajectory.append(position)    #add the position to the trajectory
        return position                     #returns current position in cartesian coordinates
    

class Animator:
    def __init__(self, pendulum, draw_trace=False):
        self.pendulum = pendulum
        self.draw_trace = draw_trace
        self.time = 0.0

        #set up the figure 
        self.fig, self.ax = plt.subplots()
        self.ax.set_ylim(-2.5,2.5)
        self.ax.set_xlim(-2.5,2.5)

        #set up window for timer
        self.time_text = self.ax.text(0.05, 0.95, '', 
                                      horizontalalignment = 'left',
                                      verticalalignment = 'top',
                                      transform = self.ax.transAxes)
        
        #plot last position of trajectory to initialize
        self.line, = self.ax.plot(
            self.pendulum.trajectory[-1][:, 0], 
            self.pendulum.trajectory[-1][:, 1], 
            marker='o')

        #trace whole trajectory of the second mass
        if self.draw_trace == True:
            self.trace, = self.ax.plot(
                [a[2, 0] for a in self.pendulum.trajectory],
                [a[2, 1] for a in self.pendulum.trajectory])
            
    def advance_time_step(self):
        while True:
            self.time += self.pendulum.dt
            yield self.pendulum.movement()
    
    def update(self, data):
        self.time_text.set_text('Elapsed Time: {:6.2f} s'.format(self.time))
        self.line.set_ydata(data[:, 1])
        self.line.set_xdata(data[:, 0])

        if self.draw_trace == True:
            self.trace.set_xdata([a[2,0] for a in self.pendulum.trajectory])
            self.trace.set_ydata([a[2,1] for a in self.pendulum.trajectory])

        return self.line
    
    def animate(self):
        self.animation = animation.FuncAnimation(self.fig, self.update, self.advance_time_step, interval = 25, blit = False, save_count=1000, cache_frame_data=False)
        self.animation.save('animation.gif', writer= 'pillow', fps = 30)

pendulum = Pendulum(theta1 = 2* np.pi - 0.1, theta2= 2 * np.pi - 0.2 , dt=0.01)
animator = Animator(pendulum=pendulum, draw_trace=True)
animator.animate()
plt.show()



    

