NLTK Tutorial

Major steps in doing language analysis
1. Getting the data
2. Turning the data into numbers
3. Analyzing the data 

This session focuses on 2. Turning the data into numbers.
There are lots of ways to do this, both programmatic and non-programatic.

Q--> What parts of language (spoken, written, texted) can you count? What numbers can you come up with beyond counting?

Before we cover this, we have to talk about data types
	* strings - anything can be a string. Think of this like a sequence of characters. Python can't look inside unless you tell it to. 
	my_string = "I am a Digital Researcher!"	
	* lists - just as it says, this is a list of items. Python can see the things in the list, but not inside the things in the list. Can add things to the list with '+' - so long as it is also a list. Otherwise, "append"
	my_list = [] (makes an empty list)
	love_list = [love, amor, te quiero, te amo, tk, tkm, ily]
	* dictionaries - key/value pair. Useful for keeping tallies of things: i.e., number of times a word appears i.e., {a:200, b:15, c:135 ... }
	my_dict = {} (makes an empty dictionary)

All the things you can look at with text analysis
	1. Text Editor Functions
		a. Word counts
		b. Word-in-context (text wrangler & sublime)

Let's get started with Python
        NLTK Functions

import nltk
from nltk.book import *

		
		i. Concordance - shows the context that the word occurs in
			text1.concordance('WORD')
			
		ii. Similar - shows words that appear near similar words **very illuminating if looking at 2 different texts - how does one person use "love" versus another"?
			text1.similar('WORD')
			
		iii. Common contexts - used to compare two words - in what environments do they both occur?  
Q--> The syntax changed here - we had to use brackets when we gave it the words to look for - WHY?!?
			text1.common_contexts([WORD, WORD]) 
			
		iv. lexical dispersion plot - good for plotting word use over time or throughout the course of a book
			text1.dispersion_plot(['hope','justice','freedom')]
		** A pop-up will appear with the dispersion plot. You can save this if you want. This MUST BE CLOSED TO MOVE ON
		
		v. Count a specific word - how many times does this sequence of characters occur in my document?
			text1.count("love")
		
		vi. Count tokens - tokens are sequences (i.e., words and punctuation, so "love", "bowie", "Bowie", "!" and ":)" are tokens)
			len(text1)
			
		vi. Count unique words - first have to make a set that groups all the "words" together (numbers, punctuation sequences, etc.) - this groups together types. Token = instance, Type = more general ("bowie" and "Bowie" are different types)
		
			1. make a set of words 
				set(text1) 
				(you might want to sort this set if you want to organize it)
				sorted(set(text1)
			2. count how many items are in that set 
				len(set(text1)))
			
		vii. Lexical Density! We have a number!! The number of unique tokens divided by the total number of words. This is a descriptive measure of language register.
			1. len(set(text1))/len(text1)
			
		viii. Frequency Distribution! We have another number!!
			1. first make an object that Python can look at:
				my_dist = FreqDist(text1)
					I like to check if its there
					type(my_dist)
				
			2. Tell python to plot this on a graph so we can inspect it
				my_dist.plot(50,cumulative=False)
				
			3. Make a dictionary of the 100 most common words and how often they occur
				my_dist.most_common(100) - gives the top 100 words and their numbers
			
We could go on and on with things you can do. 
			
	c. Pythonic non-NLTK Functions

Let's say I am interested in words that I think are related to love and I want to check if they occur in the 
        ii. Intro sentiment analysis - the love text example 
        	love_list = []
			1. make a list of words you want to use 
				love_words = ['love', 'joy', 'hope', 'amor']
			2. loop through all the words in your corpus
				for word in love_words:
					if word in text1:
						love_list.append(word)
					else:
						pass
			3. Check to see what you got!
				print(love_list)
					
	d. Regular Expressions - used for pattern matching and data cleaning. It is worth mention, but not going to discuss.
	
	
3. Your own text
	a. There are many corpus readers to make this easier, but we are going to go ste-by-step to develop an understanding of what is going on.
	
	b. Making an NLTK Text
		i. From the internet
			from urllib.request import urlopen
			url = u"http://www.gutenberg.org/cache/epub/996/pg996.txt"
			raw = urlopen(url).read()
			raw = raw.decode("utf-8", "ignore")
			
That is all that you SHOULD have to do, but if you just do this, you will get an ASCII Error. Python can't deal with non-English characters, so this is a little piece of code to fix it. For more on that and how it relates to Python, visit the Python docs here. For the purpose of this workshop, put in this last piece of code.
			Check to be sure it's right
			type(raw)
			
		ii. Now need to break this giant string into a list of things we recognize - tokens (words, punctuation, etc.)
			tokens = nltk.word_tokenize(raw)  
			
		iii. this makes a list of tokens - let's check to make sure its correct 		
			tokens[:10]
			
		iv. This is enough to read it in to make your own files, but to use the NLTK features, need to make it an NLTK Text
			dq_text = nltk.Text(tokens)
			
		v. Check to be sure it worked
			type(dq_text)
			 		
		vi. Now let's use our favorite new techniques.
		
		vii. From your own files exactly the same, just have to read the file in.
			1.  Most basic
				infile = open("PATH", "r")
				my_file = infile.read()

From here, the whole word is open to you... For more information and ways to work with text, refer to the NLTK Book! 
If you start working in Python to do text analysis, come to the PUG or Office Hours - there is so much more to learn!!
