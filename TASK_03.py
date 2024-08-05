import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = TFGPT2LMHeadModel.from_pretrained(model_name)

train_data = [
    "The quick brown fox jumps over the lazy dog.",
    "This is a sample sentence.",
    "Another example of a sentence.",
]

input_ids = tokenizer(train_data, return_tensors="tf", padding=True, truncation=True)["input_ids"]

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

for epoch in range(10):
    for batch in range(len(input_ids)):
        with tf.GradientTape() as tape:
            outputs = model(input_ids[batch], training=True)
            loss = loss_fn(input_ids[batch], outputs.logits)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        print(f"Epoch: {epoch}, Batch: {batch}, Loss: {loss.numpy()}")

model.save_pretrained("fine_tuned_gpt2")

input_text = "The quick brown fox"
input_ids = tokenizer(input_text, return_tensors="tf")["input_ids"]
outputs = model(input_ids)
generated_text = tokenizer.decode(outputs.logits.argmax(axis=-1).numpy()[0], skip_special_tokens=True)
print(generated_text)