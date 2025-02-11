PROMPT_ZERO_SHOT = """Briefly summarize the review related to {aspect}: {text}.
Please use the language used in reviews"""
# PROMPT_ZERO_SHOT = """
# Please use the language used in reviews.
# Summary example 1:If you have your heart set on a 4K Samsung TV but not a high price tag, this 50-inch model is worth considering. Features 4K resolution at a reasonable price. Smart technology is responsive and makes streaming easy. 120Hz refresh rate is appreciated by avid gamers who like fast action. Setup is simple, and the remote is straightforward to operate. The 50-inch screen is somewhat smaller than others we considered, and viewing angles could be better. But for the price, you may not mind
# Summary example 2:Delivers beautiful images and responsive smart features in a modern-looking 50-inch model. Offers smart technology that makes streaming your favorite apps easy. Setup isn't too difficult. Picture quality offers excellent contrast and a motion rate that's perfect for action movies and sports as well as gaming. Slim build. Compromised side viewing angles. Sound is tinny and lacks balance and bass
# Summary example 3: Offers almost everything viewers love in a TV in a streamlined 49-inch model – great picture, convenient features, and smart technology. Samsung's updated 49-inch smart TV delivers exceptionally vivid 4K UHD images with vibrant colors. Ultra-thin design fits nicely in most spaces. Pairs easily with your phone for smart control. Three HDMI inputs and 120 Hz refresh for your gaming needs. Setup can be challenging. Remote that comes with it isn't smart, but since control can be activated with a smartphone, this isn't a deal-breaker
# Briefly summarize the review: {text}. 
# Please use the language used in reviews"""

PROMPT_FEW_SHOT = """
Briefly summarize the ### Review related to {aspect}.
Output example: {example_summary}. 
### Review {text}.
"""

PROMPT_COT = """
1. Aspect extraction: Identify the aspect reviewed
2. Popular opinion: what opinions appear the most in reviews
3. Feature and Opinion extraction: what features is reviewed, and what is the opinion about it
4. Summarization: combine all the information into a cohesive summary like a normal review
Briefly summarize the reviews related to {aspect}: {text}.
"""

# params: text
PROMPT_EXTRACTION = {
    "amasum": """Your task is to extract entities explicitly reviewed (noun), their aspects, and expression phrases (positive or negative adjective) respectively in the following ### Review. 
Please follow template:
#Aspect name: #Entity name: #Opinion phrases: #Description:
For example:
#Aspect name: Material, Durability #Entity name: Boots #Opinion phrases: excellent, durable #Description: My boots are made of excellent leather and very durable. 
### Review: {reviews}""",
    "space": """Your task is to extract entities explicitly reviewed (noun), their aspects (just one in: Rooms, Location, Service, Cleanliness, Building, Food, General), and expression phrases (positive or negative adjective) respectively in the following ### Review.
Please follow template:
#Aspect name: #Entity name: #Expression phrases: #Description:
For example:
#Aspect name: Service #Entity name: Staff #Expression phrases: kind, friendly #Description: The staff was kind and friendly.
### Review: {reviews}"""
}

PROMPT_SUMMARIZATION = """Briefly summarize:
{knowledge_graph}
For example:
{descriptions}
"""

N_SHOT_EXAMPLES = {
    "amazon": {
        "general": [
            "These tights are great. They are durable and do not tear easily, they can be worn and washed without worry. The bottoms of can be pulled up easily so that sandals can be worn with them. It might be a good idea to order a size bigger because they can be a little tight in the waist. Overall, these tights are definitely recommended.",
            "These transition tights are perfect for children sensitive to the tight sensation other tights have around the foot.  The material is soft and durable; they stand up well to both the rough nature of children, and the washing machine.  This product does tend to run slightly small, so purchasing one size up is recommended.",
            "Bought these for my 3 year old daughter for her classes and they turned out great and fit perfectly. She can pull them up to walk in then pull down to cover her toes for class. Strong tights that should last. You might need to buy a size up just for comfort around the waist.",
            "What an impressive Thomas the Train costume! It will fit any train loving toddler for at least a few Halloween seasons due to its roomy design. The candy pocket on the front is an adorable touch. And it's so stinkin' cute! Absolutely worth the price.",
            "This Thomas costume is really cute plus it is comfortable and fits well. My child liked that it has a pocket and a hat. It even has a candy catcher. This costume fits well for a two or three year old. The hat is thin but it is completes the costume. My child loved being Thomas.",
            "This Thomas the Tank Engine Halloween costume looks very cute, and fits well for children around two to five years old. It comes with a large, built-in pouch for hands-free candy hauling. The only potential issue is the fabric seems a bit thin and flimsy. Nonetheless, I would recommend this."
        ]
    },
    "amasum": {
        "general": [
            "A small, inexpensive keyboard that can be bought in one of seven colors. An excellent portable choice if you’re on a tight budget. Incredibly lightweight at only 0.3 pounds. Small enough to easily fit in a backpack or briefcase. Bluetooth connectivity works well. Compatible with four of the major operating systems. Battery lasts a long time. Battery indication light doesn’t come on until the batteries nearly depleted", 
            "A wireless keyboard that's easy to use without needing to stock up on batteries. Keys are backlit and the brightness can be adjusted. Rechargeable with included micro USB cable. You can get up to 10 days with one charge. Offers quiet, comfortable typing. A full charge takes 3-6 hours. Difficult to clean. Keys are delicate and easy to break",
            "Offers unbeatable comfort. Feels professional, sturdy, and advanced. Simply the best key board on the market right now. The \"PerfectStroke\" key system minimizes typos and provides comfort and nearly silent typing. The slick design is paired with a touch-activated backl ight: simply waving a hand will turn it on. Doesn't boast curves like other ergonomic models. Relatively short battery life, but there is an easy USB-powered recharging system",
            "The Logitech Wireless Illuminated Keyboard K800 is a worthwhile refresh with updates that include a Unifying receiver for consolidated USB port access, rechargeable AA NiMH batteries, a nd Logitech's comfortable PerfectStroke key design. If you do your typing in the dark, the adjustable backlit Wireless K800 won't disappoint. NiMH batteries recharge with Micro-USB c able. automatic and adjustable backlit keys. Unifying receiver connects multiple devices using single plug. PerfectStroke key system offers uniform tactile feedback. Thin profile all ows sacrifices design for durability",
            "This is a great feature-filled option if money is your prime consideration, but it lacks a bit in size and power. Comes with a 12.5-inch, full 1080p touchscreen. Has a 64GB har d drive and 4GB RAM. Weighs 2.65 pounds. Attractive price. Battery is rated up to 10 hours. Boots up quickly. Backlit illuminated keyboard. 100GB free storage with Google Drive inclu ded. Chrome operating system. Only has 4GB RAM, and only two USB-C ports. Some users report problems with this computer's battery and power (stops charging or booting up within a few months)"
        ]
    },
    "space":{
        "general": [
        "The staff are friendly and exceptional. Every room (lobby included) was very clean. They are spacious, very quiet, and come with a coffee maker. Though, the rooms are outdated in decor. The hotel itself is conveniently close to the airport and restaurants. There's a chocolate-chip cookie at arrival, and for the prices, the experience is a good value.",
        "Service was exceptional and the quality was great! The rooms are always clean, quiet and spacious with nicely appointed bathrooms. The location is across the street from the airport, was within walking distance to a Denny's and other restaurants. The hotel interior itself is a bit outdated, but the room we stayed was modern.",
        "All the staff was exceptionally helpful, courteous, and friendly, keeping the rooms clean and well-prepared. The interior of the hotel needs updating, but the rooms themselves were very spacious, modern, and comfortable to stay in. The hotel itself is conveniently located near the airport, a steak restaurant, fast food, and has a free shuttle service for broader access to Seattle.",
        "The staff was friendly, helpful, and very quick. The hotel, rooms, and pool were very clean, but the tables at the Hilton Honors lounge were not cleaned well. Our room was nicely decorated, with great linens, a flat screen TV, and a beautiful view of the Miami skyline. The restaurant on the main level was excellent, with a good a variety of food. The location was good, very close to the Bayside area and just a $15 ride to Lincoln Rd/South Beach. The express check out on the phone saved a lot of time.",
        "Excellent service! The staff were pretty friendly, warm and professional. The room was very spacious and nicely put together. The bathroom is also really nice too. The restaurant on the main level was excellent, had a variety of food and drinks. The location downtown location was good, very close to the Bayside area and just a $15 ride to Lincoln Rd/South Beach. Beautiful common areas, lounge, lobby, pool.",
        "The hotel is clean and beautiful all-around and close to the Bayside area. The staff was extremely nice and attentive, from the front desk to the valet and the bellboys, always finding ways to go the extra mile. The rooms themselves were very spacious, comfortable, and clean, with nice amenities. The variety of restaurant food was excellent.",
        "The polite, professional staff is friendly and helpful with groups, and have a fine attention to detail. They keep the comfortable rooms and spacious bathrooms clean and beautiful. The cuisine the downstairs in-house restaurant serves was outstanding. The building and decor itself were both beautiful and modern.",
        "The staff was polite, accommodating, and attentive to detail. The rooms and bathrooms were very clean. They were also spacious and comfortable, with plenty of amenities. Cuisine in the restaurant is outstanding; as is room service. Overall the property has a warm, contemporary design and deserves accolades.",
        "The staff was very accommodating and professional. We were very impressed with the service and attention to detail. The rooms and bathrooms were fantastic and very clean with a large soaking tub, a shower room and excellent lighting. The master bedroom was beautiful, comfortable and had great views of downtown Dallas. The hotel features an excellent in house restaurant with plenty on the menu and delectable choices. The food was amazing! An absolutely beautiful property! The Rooms, pool, lobby, design, etc are all well deserving of the accolades they receive with their warm, modern feel."
        ],
        "rooms": [
            "The rooms are large and quite, you can't hear the planes taking off at the airport next door. The beds are comfortable and large. The bathrooms are mixed, some need cleaner doors and to be renovated, others seem clean and well appointed. The ice and vending machines are close. The coffee machine in the room is appreciated. The lighting was insufficient, and an old basement smell was present sometimes.",
            "While close to the airport, it was quiet because of thick windows. The beds were large and comfortable with lots of extra pillows. The bathrooms could use some refurbishment. Furnishings were complete with an ottoman, an easy chair, and a coffee maker. A balcony gives a great view of the surrounding city.",
            "This hotel features very comfortable and spacious rooms, with balcony, coffeemaker, comfortable beds and were well furnished. Some things that need work is the bad lighting, unkempt bathrooms and smell of mildew. All that being said, the rooms are very quiet even though the hotel is close to the airport.",
            "The rooms had excellent creature comforts, towels pillows and bedding . I felt that it was a very comfortable, nicely decorated, spacious sleeping room. The rooms overlooking the bay were fantastic with floor to ceiling windows with amazing views. The bathroom is also really nice, as well.",
            "The rooms were large and had floor to ceiling windows great views of the bay and Miami skyline. The rooms came with a pullout sofa, flat screen TV, and a refrigerator. The bed was comfortable, with fresh linens. In the bathrooms there were clean towels.",
            "The room was very large and nicely decorated. The great amenities included a flat panel TV, fridge, and two outer walls/windows with amazing views of Miami. The bathroom was also nice, with fresh-smelling towels. The bed was extremely comfortable and big, excellent for a good night's sleep.",
            "The rooms at The Joule, Dallas were spacious, beautiful, and large with a great view.",
            "The room was superb comfortable and had great views of downtown Dallas! The bathrooms are spacious and luxurious with all the amenities you could ask for.",
            "The rooms were brightly lit, had interesting photos, and had great views of downtown Dallas. Beds were comfortable. The bathroom was spacious and had a large amount of high quality amenities available."
        ],
        "location": [
            "It's a convenient location close to the airport, with shuttle service to and from the airport that runs every 15 minutes for 24 hours a day. The shuttle service is very good. It's so close you could even walk to the airport if you wanted. It's also in convenient walking distance of many restaurants.",
            "The airport was convenient to reach with the help of a speedy, twenty-four hour shuttle bus. Also located nearby, within walking distance, was a Denny's, a fast food joint, and a steak house.",
            "Within walking distance from the airport, this hotel's location is great. There is even a 24 hour shuttle that runs every 15 min that will take you to the airport or some near by places to eat like Denny's Jack in the Box and a steak place.",
            "This hotel has a very good location near the Biscayne Bay with beautiful views! It's very close to the Bayside area, and just a $15 ride to Lincoln Rd/South Beach. There were always taxi's at the hotel. There was a lot of construction around the hotel, but overall it was relaxing for downtown location.",
            "This hotel has a very good location near the Biscayne Bay and is just a $15 taxi ride to Lincoln Rd and South Beach.",
            "The location was good, not right in South Beach, but was very close to the Bayside area and near the Biscayne Bay with beautiful views! There were always taxi's at the hotel also that could bring us on a five minute trip to the beach!!",
            "The hotel is in a nice location with good access to some great restaurants. It's in a convenient location to the Dallas business hub for guests. The pool with the inifinty edge over Main St. was a great place to people watch!",
            "The hotel has great restaurants nearby, and the Dallas business hub for events, and was located right on Main St.",
            "The hotel is in a nice location with good access to some great Dallas restaurants"
        ],
        "service": [
            "The staff is exceptionally friendly and helpful both at the front desk and the restaurant. Expect sweet welcoming gifts at your check-in.",
            "Helpful, courteous, warm staff helps with a wind down after traveling. There is also a chocolate chip cookie at check-in.",
            "Mostly the staff is extremely helpful and friendly, helping to take the stress out of traveling. The cookies given at check in were greatly appreciated.",
            "The staff was always friendly, accommodating, and helpful. The valet employees are very quick and nice.",
            "The service from different staff (valet, front desk, and food) were excellent, ambivalent, and professional.",
            "The professional staff was excellent, from the front desk to the valet. Everyone as more than friendly and accommodating, giving us helpful, attentive service that goes the extra mile.",
            "The staff was extremely friendly, and helpful, from making arrangements over the phone with the concierge prior to arrival, through the time we left the hotel. They were not at all condescending, which can sometimes occur in a high-end hotel.",
            "The professional staff was extremely friendly, polite, and helpful. They work with large groups easily and are accommodating of all one's needs, paying attention to all the important details.",
            "We were very impressed with the service and attention to detail. The staff is extremely friendly, professional and helpful."
        ],
        "cleanliness": [
            "The spacious hotel lobby and rooms are very clean, comfortable, and well-appointed.",
            "Although the hotel's architecture feels dated, the rooms and bathrooms are clean.",
            "Even thought there was a minor issue with gaining access to the room because of a faulty magnetic door, the room and bedding were clean and comfortable.",
            "The rooms at Hilton Miami Downtown were very clean.",
            "The room and the bathroom were both very clean. So was the pool. The hotel seemed renovated and clean. It was hard to find a clean breakfast table at the Hilton Honors lounge though, as the supervisor was the only one who took the initiative to clean the tables.",
            "By-and-large everything from the facility to the rooms and bathrooms are very clean. The lounge is lacking in that department as far as clean tables go, though.",
            "The area was very clean and very well maintained. The rooms were very clean.",
            "The rooms were very clean, beautiful, and well maintained",
            "The hotel's rooms and bathrooms were very clean."
        ],
        "building": [
            "The historical hotel lobby were very attractive. The balcony had a great view of trees . The spa and heated pool is a kid-friendly area and also has wi-fi. There is even a laundry room available to the guests.",
            "Hotel with very nice lobby and relaxing spa/pool area with lounge and free wifi. The pool is big and kid-friendly. There is also a beautiful view of the trees from the balcony.",
            "Warm, beautiful, large pool for the family. Old fashioned interior but pleasant rooms, great balcony, and the view outside to the trees was relaxing.",
            "The lobby, pool, and gym of the hotel were all very beautiful.",
            "Hilton Miami Downtown has very nice rooms and has a bar/restaurant downstairs. There's also a swimming pool and a gym with great views.",
            "Beautiful common areas, lounge, lobby, pool. The restautant, gym and bar looked great at this hotel. Free internet is available only in lobby downstairs and if you want internet in your room, you have to pay for it.",
            "The Joule Hotel was a beautiful property with spacious rooms, a roof top pool and a five star restaurant.",
            "This is a warm, modern hotel on amazing property, with an amazing rooftop pool and lounge area. The inside is modern and sleek, while the fitness facilities are nice, with plenty of aerobic equipment and a enough weights to get in a good workout.",
            "Absolutely beautiful property, rooms are spacious and modern, and the roof-top pool area is excellent. Their fitness facilities were nice, with plenty of aerobic equipment and enough weights to get a nice workout. The inside is very modern and sleek; the room had a sexy contemporary decor."
        ],
        "food": [
            "The hotel restaurant's food was nicely presented, and sometimes good. However, sometimes it was bland and tasteless, and a bit pricey. The restaurant's clam chowder was good. The breakfast buffet isn't a bad deal for what you get. The fresh cookies given at check in were delicious.",
            "Food was well presented and some of it was tasty, if a little pricey, but the clam chowder at the restaurant and the breakfast buffet made the trip all the more worth it. Dave's Diner next door was also enjoyable.",
            "Although some of the food was bland and a little overpriced, the clam chowder was good. The staff even gave out these delicious freshly baked cookies int he reception area and the breakfast buffet is also a great value for what is offered.",
            "The breakfast was generally very good , although it was the same every day, and sometimes reported as cold. Complimentary coffee, fruit, and bottled water were offered once one exists the elevator in the morning into the lobby. The restaurant on the main level was excellent, had a variety of food and drinks. The breakfast buffet was expensive.",
            "The noninclusive continental breakfast buffet is expensive at $20, but it's worth it. Complimentary coffee, fruit, and bottled water are also available once one exits the elevator in the morning into the lobby.",
            "Complimentary coffee, fruit, and bottled water were offered at the Hilton Miami Downtown. The breakfast buffet wasn't complimentary but it was well worth the $20.00.",
            "The quality of the food was great, with plenty of delectable choices on the menu to choose from already, and it's also great for breakfast",
            "Room service from the restaurant downstairs (and the restaurant itself) is amazing. The grilled cheese being one example.",
            "The hotel has an excellent in house restaurant with plenty on the menu and delectable choices. It is also very nice for breakfast. We had dinner at Nobu which is in Crescent Court which was really good too."
        ]
    }
}

max_new_tokens ={
    'rooms': 35,
    'location': 60,
    'service': 20,
    'cleanliness': 35,
    'building': 15,
    'food': 20,
    'general': 80,
}

ASPECT_NAMES = {
    'space': ["general", "rooms", "location", "service", "cleanliness", "building", "food"],
    'amasum': ["general"]
}
