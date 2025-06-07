# Build stage
FROM golang:1.21-alpine AS builder

WORKDIR /app

# Install dependencies
RUN apk add --no-cache git ca-certificates

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o data-pipeline cmd/data-pipeline/main.go

# Final stage
FROM alpine:latest

# Install certificates
RUN apk --no-cache add ca-certificates

WORKDIR /root/

# Copy the binary from builder stage
COPY --from=builder /app/data-pipeline .

# Copy config files
COPY --from=builder /app/configs ./configs

# Create data directory
RUN mkdir -p /app/data

# Run the binary
CMD ["./data-pipeline"] 