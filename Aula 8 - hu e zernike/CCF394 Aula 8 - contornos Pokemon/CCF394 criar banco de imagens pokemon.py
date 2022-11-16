import requests
import bs4 as bs
# pip install lxml
# Get Pokemon names
NO_POKEMON = 150 # Maximum of 802
print("iniciando....")
pokedb_urls = ['https://pokemondb.net/pokedex/'+str(i+1) for i in range(NO_POKEMON)]
pokemon_names = []
for url in pokedb_urls:
    page = bs.BeautifulSoup(requests.get(url).text, 'lxml')
    pokemon_name = page.find('link', {'rel':'canonical'})['href'].split('/')[-1]
    pokemon_names.append(pokemon_name)
    print(pokemon_name)
    
# Write their names to a file for future reference
f = open('pokemon_names', 'w')
f.write(str(pokemon_names))
f.close()

# Download image of each Pokemon (full colour)
for name in pokemon_names:
    img_url = 'https://img.pokemondb.net/artwork/%s.jpg' % name
    img_bytestr = requests.get(img_url).content
    with open(('./pokemon_imgs/%s.jpg' % name), 'wb') as img_f: # Closes nicely
        img_f.write(img_bytestr)
        
# Download R&B sprites (grayscale)
for name in pokemon_names:
    img_url = 'http://img.pokemondb.net/sprites/red-blue/normal/%s.png' % name
    img_bytestr = requests.get(img_url).content
    with open(('./pokemon_rb_imgs/%s.jpg' % name), 'wb') as img_f: # Closes nicely
        img_f.write(img_bytestr)
