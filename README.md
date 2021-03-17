# channel-attribution
A plugin for Channel Attribution. Can use either Markov chains or Shapley Values.
Channel Attribution is a method for attributing outcomes to an event. 
It is commonly used in marketing to attribute conversions to a specific type of marketing interaction.
Read more on Channel Attribtion here: https://blog.dataiku.com/step-up-your-marketing-attribution-with-game-theory

There are two methods for this contained in this plugin.


# Markov Chains

Markov Chains are built using "chains" of events. Given a starting point, a Markov Model will predict what is the next most likely event to occur.

Examples of a few possible chains:  
1. A Custom Googles Dataiku > 2. The Customer Visits the Website > 3. The Customer Send Us An Email
1. A Custom Googles Dataiku > 2. The Customer Send Us An Email
1. A Custom Googles Dataiku > 2. The Customer Visits the Website > 3. The Customer Googles A Competitor

The Markov Chain will tell us the probability that a customer who has just visited the website is about to send us an email. 
It will calculate a probability like this for every possible combination of a Stating Point and a Next Step.


# Shapley Values

Uses Shapley values to attribute an outcome to a feature. Shapley values measure the marginal contribution of an event towards an outcome.

