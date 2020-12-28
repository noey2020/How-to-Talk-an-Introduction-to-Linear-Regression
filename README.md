# How-to-Talk-an-Introduction-to-Linear-Regression

December 27, 2020

I appreciate comments. Shoot me an email at noel_s_cruz@yahoo.com!

Hire me! 😊

- For the past few weeks we've been focusing
on classification problems.
What we'll do today is to switch gears a little bit
and look at regression.
Today, we'll look at a one-dimensional version
of the problem as an introduction,
and this will allow us to define some basic concepts
like predictor and response variables.
We'll see how this problem can be formulated
as an optimization task,
and how it can be solved using elementary calculus.
So in one dimension, regression is simply the business
of fitting a line to a bunch of points.
This is something we've all seen before.
We have a x-axis and a y-axis.
We have a set of points.
And we wanna fit a line to them.
There's no line that goes exactly through the points,
and so we just want something that kind of
goes through the middle of them somehow.
How exactly do we do this?
And why would we want to do this?
So let's look at a little example over here.
A certain Ivy League university
collected data on its entering freshman class.
So at the end of freshman year,
they noted down everybody's GPA.
That's the histogram you see over here.
So the GPA, the grade point average,
is a number between zero and four.
That's what's shown on the horizontal axis.
And on the vertical axis is the frequency,
the number of people who got that grade.
In this case, it looks like the most frequent grade
was somewhere over here.
So something like 2.25, which is a C plus.
So here's a question.
Suppose a random student shows up
and we want to predict his or her GPA.
What score would we predict?
Given this information, a reasonable thing to do
would be to simply predict the mean of this distribution.
That's easy enough to compute.
If we do that, it turns out that in this case it is 2.47.
So how good is this prediction?
One way to sort of assess its quality is to say,
"What is its average squared error?"
So a student has shown up at random, okay?
So this is their actual GPA.
And we predicted this particular value, 2.47.
If we look at the difference between these two
and square it,
what's the expected value of that?
Well, if we look at this closely,
this is exactly the formula for variance.
It is the variance of this distribution.
And in this case, the variance turns out to be .55.
Now, it turns out that at this university,
they had also collected some other information.
In addition to having everybody's freshman year GPA,
they also had the SAT score from high school.
This is a scatter plot of the two against each other.
So on the horizontal axis are the SAT scores,
and on the vertical axis are the college GPAs.
You can see that the data is tilted slightly upwards,
which indicates a positive correlation.
So what we can do in this case is to see
whether the SAT score helps us to predict the college GPA.
So we go ahead and we fit a line like this.
Now, we'll get to the business of exactly
how we fit this line,
but for the time being, let's just see how we would use it.
Let's say a random student shows up,
and we wanna predict his or her GPA.
Well, now we find out what the SAT score is.
And let's say it's 1200.
The SAT score is 1200.
We just go up to this line,
we go across, and our prediction is three.
We predict a GPA of three.
Now, can we use this additional piece of information
to make predictions?
It turns out that the mean squared error drops.
It drops to about .43.
So that additional piece of information
was actually quite helpful.
This is a classic regression problem.
There's something we want to predict,
which is called the response variable,
that's the college GPA,
and then there's information
that helps us make this prediction.
Those are the predictor variables,
and in this case, there's just one predictor variable,
the SAT score.
So this is the x, the thing that helps us make predictions.
And then there's the value we actually want to guess,
which is the y, the college GPA.
So what is the mean squared error though?
Well, let's look at this data, okay?
So look at this person over here.
Using our line, the prediction we would make for that person
would actually be down here.
So there's a certain amount of error.
There's this much error, and we can square that.
Let's look at that squared error, okay?
Let's look at this person over here.
The squared error on that person is this distance squared.
Let's look at this one over here.
The squared error on that person is this distance squared,
it's much smaller.
Let's look at this person.
The squared error on their squared error
is actually pretty large.
So if we take the average of all these squared errors,
that is the mean squared error.
So, how do we go about fitting a line of this form?
Well, the first thing is to figure out
how you're going to parametrize a line.
And one convenient way of doing this
is to write down a line as y equals ax plus b,
where a is the slope of the line and b is the y-intercept.
For example, let's say that we have
our x-axis and our y-axis,
and we have a line like this.
Maybe this is two and this is one.
So the y-intercept is where the line crosses the y-axis,
that's one.
And the slope in this case, well it's sloped downwards,
so it's gonna be negative.
And the slope in this case is negative 1/2.
And so the equation of this line
is y equals negative 1/2 x plus one.
So we would parametrize a line using these two numbers,
a and b.
Okay.
So this lets us define the problem precisely.
We're given a set of data points.
These are x, y pairs.
X and y are both real numbers.
The x are the predictor variables,
in our case, the SAT scores.
The Ys are the things we wanna guess,
the response variables, in our case, the GPAs.
So we have n data points of this form.
We wanna find a line.
In other words, we wanna find parameters a and b
that define a line.
And what we want is to find the line
that incurs the least mean squared error.
So what is that?
That's what's given by this formula over here.
If we look at the ith data point, x sub i,
then using this line a, b,
our prediction on x sub i would be this.
That would be the value we would guess.
The correct value is y sub i.
So the squared error incurred on the ith data point
is this term over here.
Now we take that and we average over all the data points.
This is the mean squared error.
And this is the loss function
that we are trying to minimize.
There's something interesting going on over here.
We are taking a learning task
and formulating it as an optimization problem,
as the problem of finding parameters
that minimize some kind of loss function.
This optimization approach
has proved to be extremely fruitful in machine learning,
and we'll be seeing a whole lot more of it.
So let's go ahead and see
how we can minimize this loss function.
So we wanna minimize this function L
which has two parameters, a and b.
Well, we can just take the derivative and set it to zero.
Usual calculus way.
So since the two parameters,
we need to take the derivative with respect to each of them.
So to minimize,
what we will do is we'll simply set
the derivative with respect to a
and the derivative with respect to b
to be zero.
Okay, so let's go ahead and compute these derivatives.
I think the one with respect to b
is gonna be a little bit simpler, so let's start with that.
What is the derivative of L with respect to b?
Okay, so L is a big summation.
The derivative of a sum is the sum of the derivatives.
So we'll start by writing that down.
Now, how do we take a derivative of that?
We have the squared term over there.
Okay, so there's this general rule which says,
okay, so we're taking the derivative with respect to b,
"If you're taking the derivative of something squared,
"u squared,
"that turns out to be two u d u."
So it's two u times the derivative of u with respect to b.
And for us, u is this thing over here.
That is the u, we wanna take the derivative of that.
So let's write down two u first of all.
So two, we just copy down u.
Okay.
And now we take the derivative of u with respect to b,
which is actually just negative one, okay?
So times negative one.
Good, so that's the entire derivative,
and we wanna set that to zero.
Now, since we're setting it to zero,
we can just divide out the negative one
and we can divide out the two.
And let's see what we get.
So we get the summation of the y Is
equals a times the summation of the x Is
plus n times b.
Because we're summing b from i equals one to n,
so that's n times b.
And let's go ahead and solve for b.
So that tells us that b is equal to,
we can divide both sides by n,
one over n times the summation of the y Is
minus a times one over n times the summation of the x Is.
That's the optimal setting for b.
So what are these things over here?
What is this term over here?
Well, that's just the average y value,
the average response value.
And what is this thing over there?
Well, that's just the average x value.
In our case case, that would average SAT score.
Okay, so let's write that down in a slightly more
compact form.
Okay, so we'll just remember this equation.
And what we said is let's look at the average x value.
I'm just gonna call that x bar, okay?
So we take the average of all the Xs.
And let's look at the average of all the Ys.
And it turned out that the optimal setting for b
is the average y value minus a times the average x value.
And this makes perfect sense.
We want y to be equal to a x plus b.
Now, of course, we aren't gonna be able to get that exactly,
but it makes perfect sense
that b would be the average y value
minus a times the average x value.
Okay, so we still have to solve for a.
And the way we do that is we set the derivative of L
with respect to a to zero.
And then we go through the calculation.
And this time the algebra is a little bit more involved,
but I can just tell you what the final answer is gonna be.
If you like you can try out the steps at home.
It's just a little bit messier.
And it turns out that what the optimal setting for a is
is simply the covariance of x and y
divided by the variance of x.
So it's something of this form.
We take all the Ys,
we subtract off the average y,
and you look at the covariance with x.
And then you divide by
the x Is minus the average x squared.
So that's the optimal setting for a.
Very simple, a nice closed-form solution.
Well, that concludes our little introduction to regression.
Today, we just talked about the one-dimensional setting
where there's a single predictor variable.
What we'll do next time is to generalize this methodology
to multiple predictor variables.
See you then.

I included some posts for reference.

https://github.com/noey2020/Hpw-to-Talk-More-Generative-Models

https://github.com/noey2020/How-to-Talk-Gaussian-Generative-Models

https://github.com/noey2020/How-to-Talk-Multivariate-Gaussian

https://github.com/noey2020/How-to-Talk-Linear-Algebra-Review-3

https://github.com/noey2020/How-to-Talk-Linear-Algebra-Review-2

https://github.com/noey2020/How-to-Talk-Linear-Algebra-Review-1

https://github.com/noey2020/How-to-Talk-2D-Generative-Modeling

https://github.com/noey2020/How-to-Talk-Probability-Review-3

https://github.com/noey2020/How-to-Talk-Probability-Review-2

https://github.com/noey2020/How-to-Talk-Generative-Modeling-in-One-Dimension

https://github.com/noey2020/How-to-Talk-Probability-Review-1

https://github.com/noey2020/How-to-Talk-Generative-Approach-to-Classification

https://github.com/noey2020/How-to-Talk-of-Fitting-a-Distribution-to-Data-

https://github.com/noey2020/How-to-Talk-of-Host-of-Prediction-Problems

https://github.com/noey2020/How-to-Talk-of-Useful-Distance-Functions

https://github.com/noey2020/How-to-Talk-of-Improving-Nearest-Neighbor

https://github.com/noey2020/How-to-Talk-of-Prediction-Problems

https://github.com/noey2020/How-to-Talk-Matlab-Tricks-and-Tweaks

https://github.com/noey2020/How-to-Talk-Trading-and-Investing

https://github.com/noey2020/How-to-Work-in-Matlab-Development-Environment

https://github.com/noey2020/How-to-Talk-Vaccines

https://github.com/noey2020/How-to-Talk-Regression-in-Matlab

https://github.com/noey2020/How-to-Get-Started-in-Matlab

https://github.com/noey2020/How-to-Convert-Data-from-Web-Service-Using-Matlab

https://github.com/noey2020/Quote-for-the-Day

https://github.com/noey2020/How-to-Talk-Good-Investment-Strategy

https://github.com/noey2020/How-to-Talk-of-Good-Plan

https://github.com/noey2020/Thought-for-the-Day

https://github.com/noey2020/How-to-Talk-Stock-Watch-of-the-Day

https://github.com/noey2020/How-to-Talk-Data-Science

https://github.com/noey2020/How-to-Talk-Fundamental-Analysis

https://github.com/noey2020/How-to-Read-Company-Profiles

https://github.com/noey2020/How-to-Import-Data-from-Spreadsheets-and-Text-Files-Matlab-Without-Coding

https://github.com/noey2020/How-to-Talk-Model-of-Stock-Market-Prices-

https://github.com/noey2020/How-to-Talk-Digital-Wallets

https://github.com/noey2020/How-to-Talk-Investing

https://github.com/noey2020/How-to-Double-Your-Money-in-5years

https://github.com/noey2020/How-to-Talk-Matlab

I appreciate comments. Shoot me an email at noel_s_cruz@yahoo.com!
